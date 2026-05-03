# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import shutil
import math
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
import torch
import os
import copy
from utils.wan_wrapper_stream_t1 import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

import requests
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from utils.debug_option import DEBUG
import torch.distributed as dist
import numpy as np
from einops import rearrange
from torchvision.io import write_video
from hpsv3 import HPSv3RewardInferencer
from metrics.VideoAlign import VideoVLMRewardInference

DEFAULT_VIDEO_FPS = 16

# selfattention的sinksize，append列表需要随branch而变化
class StreamT1CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        print(f"image reward inferencer: HPSv3 loading...")
        self.image_reward_inferencer = HPSv3RewardInferencer(checkpoint_path="metrics/models/hpsv3_model/HPSv3.safetensors",device=device)
        print(f"video reward inferencer: VideoAlign loading...")
        self.video_reward_inferencer = VideoVLMRewardInference("metrics/models/videoalign", device=device)

        # hard code for Wan2.1-T2V-1.3B
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_output_frames = args.num_output_frames
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.beam_size = getattr(args, "beam_size", 1)
        self.top_k = getattr(args, "top_k", 1)
        self.seed = getattr(args, "seed", 42)
        self.sliding_window_size = getattr(args, "sliding_window_size", 10)
        # self.long_threshold = getattr(args, "long_threshold", 0.1)
        # self.short_threshold = getattr(args, "short_threshold", 0.1)
        # self.noise_fusion_sigma = getattr(args, "noise_fusion_sigma", 1)
         
        self.output_folder = args.output_folder
        self.local_attn_size = args.model_kwargs.local_attn_size
        self.sink_size = args.model_kwargs.sink_size
        self.max_sink_size = args.model_kwargs.max_sink_size

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    
    def generate_candidate(
            self,
            noise_shape: Tuple[int, int, int, int, int],
            conditional_dict: Any,
            beam: Dict[str, Any],
            block_idx: int,
            current_start_frame: int,
            candidate_seed: int,
            text_prompt: str,
            output_folder: str,
            dtype: torch.dtype,
            device: torch.device,
            short_threshold: float,
            long_threshold: float,
            noise_fusion_sigma: float,
            short_score_fusion_threshold: float,
            beam_idx: str = "",
    ) -> Dict[str, Any]:
        with torch.no_grad():
            branch_generator = self._build_branch_generator(candidate_seed, device)
            local_kv_cache = copy.deepcopy(beam["kv_cache"])
            batch_size, current_num_frames, num_channels, height, width = noise_shape

            noisy_input = self._build_candidate_noise(
                beam=beam,
                block_idx=block_idx,
                batch_size=batch_size,
                current_num_frames=current_num_frames,
                num_channels=num_channels,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                branch_generator=branch_generator,
                noise_fusion_sigma=noise_fusion_sigma,
            )
            beam_noise = noisy_input.clone()
            reward_score = copy.deepcopy(beam["reward_score"])

            denoised_pred = self._run_denoising_loop(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                local_kv_cache=local_kv_cache,
                current_start_frame=current_start_frame,
                reward_score=reward_score,
                batch_size=batch_size,
                current_num_frames=current_num_frames,
                dtype=dtype,
                device=device,
                branch_generator=branch_generator,
            )

            new_output = beam["output"].clone()
            new_output[:, current_start_frame: current_start_frame + current_num_frames] = denoised_pred.to(beam["output"].device)

        self.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=torch.ones_like(denoised_pred[:, :, 0, 0, 0], dtype=torch.long, device=device) * self.args.context_noise,
            kv_cache=local_kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length,
            reward_score=reward_score,
        )

        short_score = self._decode_short_candidate_video(
            denoised_pred=denoised_pred,
            text_prompt=text_prompt,
            output_folder=output_folder,
            block_idx=block_idx,
            beam_idx=beam_idx,
            candidate_seed=candidate_seed,
            device=device,
        )

        long_score = self._decode_long_candidate_video(
            new_output=new_output,
            current_start_frame=current_start_frame,
            current_num_frames=current_num_frames,
            output_folder=output_folder,
            block_idx=block_idx,
            beam_idx=beam_idx,
            candidate_seed=candidate_seed,
            text_prompt=text_prompt,
            device=device,
        )

        reward_score[str(current_start_frame)] = {
            "block_idx": block_idx,
            "short_score": short_score,
            "long_score": long_score,
        }

        self._update_reward_history(
            reward_score=reward_score,
            block_idx=block_idx,
            current_start_frame=current_start_frame,
            short_score=short_score,
            long_score=long_score,
            short_threshold=short_threshold,
            long_threshold=long_threshold,
            local_kv_cache=local_kv_cache,
        )

        final_score = self._compute_hybrid_score(
            short_score=short_score,
            long_score=long_score,
            current_start_frame=current_start_frame,
            current_num_frames=current_num_frames,
            short_score_fusion_threshold=short_score_fusion_threshold,
        )

        print(
            f"short score: {short_score}, long score: {long_score}, final hybrid score: {final_score} = "
            f"{(current_start_frame / (self.num_output_frames - current_num_frames)) if self.num_output_frames != current_num_frames else 0:.4f} * {short_score} + {1 - ((current_start_frame / (self.num_output_frames - current_num_frames)) if self.num_output_frames != current_num_frames else 0):.4f} * {long_score}"
        )

        del denoised_pred
        torch.cuda.empty_cache()

        return {
            "output": new_output,
            "kv_cache": local_kv_cache,
            "history_seeds": beam["history_seeds"] + [candidate_seed],
            "score": final_score,
            "noise": beam_noise,
            "reward_score": reward_score,
        }

    def _build_branch_generator(self, candidate_seed: int, device: torch.device) -> torch.Generator:
        branch_generator = torch.Generator(device=device)
        branch_generator.manual_seed(candidate_seed)
        return branch_generator

    def _build_candidate_noise(
            self,
            beam: Dict[str, Any],
            block_idx: int,
            batch_size: int,
            current_num_frames: int,
            num_channels: int,
            height: int,
            width: int,
            device: torch.device,
            dtype: torch.dtype,
            branch_generator: torch.Generator,
            noise_fusion_sigma: float,
    ) -> torch.Tensor:
        sample_noise = torch.randn(
            batch_size,
            current_num_frames,
            num_channels,
            height,
            width,
            device=device,
            dtype=dtype,
            generator=branch_generator,
        )

        if block_idx == 0:
            return sample_noise

        return noise_fusion_sigma * beam["noise"] + math.sqrt(1 - noise_fusion_sigma**2) * sample_noise

    def _run_denoising_loop(
            self,
            noisy_input: torch.Tensor,
            conditional_dict: Any,
            local_kv_cache: List[Dict[str, Any]],
            current_start_frame: int,
            reward_score: Dict[str, Any],
            batch_size: int,
            current_num_frames: int,
            dtype: torch.dtype,
            device: torch.device,
            branch_generator: torch.Generator,
    ) -> torch.Tensor:
        denoised_pred = None

        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=device,
                dtype=torch.int64,
            ) * current_timestep

            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=local_kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                reward_score=reward_score,
            )

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn(
                        denoised_pred.flatten(0, 1).shape,
                        device=device,
                        dtype=dtype,
                        generator=branch_generator,
                    ),
                    next_timestep * torch.ones(
                        [batch_size * current_num_frames],
                        device=device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

        return denoised_pred

    def _decode_short_candidate_video(
            self,
            denoised_pred: torch.Tensor,
            text_prompt: str,
            output_folder: str,
            block_idx: int,
            beam_idx: str,
            candidate_seed: int,
            device: torch.device,
    ) -> float:
        short_tensor_video = denoised_pred.clone()

        with torch.no_grad():
            video = self.vae.decode_to_pixel(short_tensor_video, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = rearrange(video, "b t c h w -> b t h w c").cpu()
            self.vae.model.clear_cache()

            short_output_folder = os.path.join(output_folder, str(block_idx))
            if beam_idx:
                short_output_folder = os.path.join(short_output_folder, beam_idx)
            short_output_folder = os.path.join(short_output_folder, "short")
            os.makedirs(short_output_folder, exist_ok=True)

            image_paths: List[str] = []
            image_dir = os.path.join(short_output_folder, f"{candidate_seed}")
            os.makedirs(image_dir, exist_ok=True)

            frames = (255.0 * video).numpy().astype(np.uint8)[0]
            for i, frame in enumerate(frames):
                img_path = os.path.join(image_dir, f"frame_{i:04d}.png")
                Image.fromarray(frame).save(img_path)
                image_paths.append(img_path)

            prompts = [text_prompt] * len(image_paths)
            rewards = self.image_reward_inferencer.reward(prompts=prompts, image_paths=image_paths)
            short_scores = [reward[0].item() for reward in rewards]
            return float(np.mean(short_scores))

    def _decode_long_candidate_video(
            self,
            new_output: torch.Tensor,
            current_start_frame: int,
            current_num_frames: int,
            output_folder: str,
            block_idx: int,
            beam_idx: str,
            candidate_seed: int,
            text_prompt: str,
            device: torch.device,
    ) -> float:
        long_tensor_video = new_output[:, max(0, current_start_frame - (self.sliding_window_size - 1) * self.num_frame_per_block): current_start_frame + current_num_frames]

        with torch.no_grad():
            video = self.vae.decode_to_pixel(long_tensor_video.to(device), use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = rearrange(video, "b t c h w -> b t h w c").cpu()
            self.vae.model.clear_cache()

            long_output_folder = os.path.join(output_folder, str(block_idx))
            if beam_idx:
                long_output_folder = os.path.join(long_output_folder, beam_idx)
            long_output_folder = os.path.join(long_output_folder, "long")
            os.makedirs(long_output_folder, exist_ok=True)
            output_path = os.path.join(long_output_folder, f"{candidate_seed}.mp4")
            write_video(output_path, video[0], fps=DEFAULT_VIDEO_FPS)
            return self.video_reward_inferencer.reward([output_path], [text_prompt], use_norm=True)[0]["Overall"]

    def _update_reward_history(
            self,
            reward_score: Dict[str, Any],
            block_idx: int,
            current_start_frame: int,
            short_score: float,
            long_score: float,
            short_threshold: float,
            long_threshold: float,
            local_kv_cache: List[Dict[str, Any]],
    ) -> None:
        if block_idx == 0:
            return

        sorted_keys = sorted(reward_score.keys(), key=lambda k: int(k))
        history_keys = sorted_keys[:block_idx]
        recent_history_keys = history_keys[max(0, 1 - self.sliding_window_size):]
        recent_short_scores = [reward_score[i]["short_score"] for i in recent_history_keys]
        average_short = sum(recent_short_scores) / len(recent_short_scores)

        prev_key = history_keys[-1]
        prev_long = reward_score[prev_key]["long_score"]
        long_drop = prev_long - long_score
        is_long_low = long_drop > long_threshold
        short_increase = short_score - average_short
        is_short_high = short_increase > short_threshold

        reward_score[str(current_start_frame)]["ema"] = is_short_high and not is_long_low
        reward_score[str(current_start_frame)]["append"] = is_short_high and is_long_low

        print(f"block_idx {block_idx} Candidate seed {current_start_frame}")
        print(f" long_drop : {long_drop}")
        print(f"is_short_high={is_short_high}, is_long_low={is_long_low}, ema={reward_score[str(current_start_frame)]['ema']}, append={reward_score[str(current_start_frame)]['append']}")
        print(f"current_sink_size:{local_kv_cache[0]['current_sink_size']}")
        print(f"current_kv_cache_size:{local_kv_cache[0]['current_kv_cache_size']}")
        print(f"append_sink_start_frame:{local_kv_cache[0]['append_sink_start_frame']}")
        print(f"global_end_index:{local_kv_cache[0]['global_end_index']}")
        print(f"local_end_index:{local_kv_cache[0]['local_end_index']}")

    def _compute_hybrid_score(
            self,
            short_score: float,
            long_score: float,
            current_start_frame: int,
            current_num_frames: int,
            short_score_fusion_threshold: float,
    ) -> float:
        alpha = current_start_frame / (self.num_output_frames - current_num_frames)
        if alpha > short_score_fusion_threshold:
            alpha = short_score_fusion_threshold
        return alpha * short_score + (1 - alpha) * long_score

    def inference(
        self,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        idx: Optional[int] = None,
        short_threshold: Optional[float] = None,
        long_threshold: Optional[float] = None,
        noise_fusion_sigma: Optional[float] = None,
        short_score_fusion_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_output_frames, num_channels, height, width = 1, self.num_output_frames, 16, 60, 104
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=dtype,
        )

        init_events = self._setup_profiling(profile)

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        sink_size = getattr(self.args.model_kwargs, "sink_size", 0)
        max_sink_size = getattr(self.args.model_kwargs, "max_sink_size", 0)

        self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size + max_sink_size - sink_size)

        if profile:
            init_events["init_end"].record()
            torch.cuda.synchronize()
            init_events["diffusion_start"].record()

        beams = [self._build_initial_beam(output)]
        result_dir = os.path.join(self.output_folder, f"{idx}")
        os.makedirs(result_dir, exist_ok=True)

        all_num_frames = [self.num_frame_per_block] * num_blocks
        for block_idx, current_num_frames in enumerate(all_num_frames):
            if profile:
                init_events["block_start"].record()

            print(f"Processing Block {block_idx} (Frames {current_start_frame} - {current_start_frame + current_num_frames})")

            if block_idx == 0:
                beams = self._generate_initial_candidates(
                    conditional_dict=conditional_dict,
                    result_dir=result_dir,
                    beam=beams[0],
                    current_start_frame=current_start_frame,
                    block_idx=block_idx,
                    current_num_frames=current_num_frames,
                    num_channels=num_channels,
                    height=height,
                    width=width,
                    dtype=dtype,
                    device=device,
                    text_prompt=text_prompts[0],
                    short_threshold=short_threshold,
                    long_threshold=long_threshold,
                    noise_fusion_sigma=noise_fusion_sigma,
                    short_score_fusion_threshold=short_score_fusion_threshold,
                )
            else:
                beams = self._generate_followup_candidates(
                    conditional_dict=conditional_dict,
                    beams=beams,
                    result_dir=result_dir,
                    current_start_frame=current_start_frame,
                    block_idx=block_idx,
                    current_num_frames=current_num_frames,
                    num_channels=num_channels,
                    height=height,
                    width=width,
                    dtype=dtype,
                    device=device,
                    text_prompt=text_prompts[0],
                    short_threshold=short_threshold,
                    long_threshold=long_threshold,
                    noise_fusion_sigma=noise_fusion_sigma,
                    short_score_fusion_threshold=short_score_fusion_threshold,
                )

            print(
                f"After Block {block_idx}, top {self.top_k} candidates have scores: "
                f"{[beam['score'] for beam in beams]} and seeds: {[beam['history_seeds'] for beam in beams]}"
            )

            if profile:
                init_events["block_end"].record()
                torch.cuda.synchronize()
                block_time = init_events["block_start"].elapsed_time(init_events["block_end"])
                init_events["block_times"].append(block_time)

            current_start_frame += current_num_frames

        if profile:
            init_events["diffusion_end"].record()
            torch.cuda.synchronize()
            diffusion_time = init_events["diffusion_start"].elapsed_time(init_events["diffusion_end"])
            init_time = init_events["init_start"].elapsed_time(init_events["init_end"])
            init_events["vae_start"].record()

        best_video, best_score = self._write_final_videos(beams, text_prompts[0], idx, device)
        self._write_best_video_metadata(result_dir, best_video, best_score, text_prompts[0])

        shutil.rmtree(result_dir)

        if profile:
            init_events["vae_end"].record()
            torch.cuda.synchronize()
            vae_time = init_events["vae_start"].elapsed_time(init_events["vae_end"])
            total_time = init_time + diffusion_time + vae_time
            self._print_profile_results(init_time, diffusion_time, vae_time, total_time, init_events["block_times"])

        # if return_latents:
        #     return video, output.to(noise.device)
        # else:
        #     return video

    def _setup_profiling(self, profile: bool) -> Dict[str, Any]:
        if not profile:
            return {}

        init_start = torch.cuda.Event(enable_timing=True)
        init_end = torch.cuda.Event(enable_timing=True)
        diffusion_start = torch.cuda.Event(enable_timing=True)
        diffusion_end = torch.cuda.Event(enable_timing=True)
        vae_start = torch.cuda.Event(enable_timing=True)
        vae_end = torch.cuda.Event(enable_timing=True)
        block_times: List[float] = []
        block_start = torch.cuda.Event(enable_timing=True)
        block_end = torch.cuda.Event(enable_timing=True)
        init_start.record()

        return {
            "init_start": init_start,
            "init_end": init_end,
            "diffusion_start": diffusion_start,
            "diffusion_end": diffusion_end,
            "vae_start": vae_start,
            "vae_end": vae_end,
            "block_start": block_start,
            "block_end": block_end,
            "block_times": block_times,
        }

    def _build_initial_beam(self, output: torch.Tensor) -> Dict[str, Any]:
        return {
            "output": output,
            "kv_cache": self.kv_cache1,
            "history_seeds": [],
            "reward_score": {},
        }

    def _generate_initial_candidates(
            self,
            conditional_dict: Any,
            result_dir: str,
            beam: Dict[str, Any],
            current_start_frame: int,
            block_idx: int,
            current_num_frames: int,
            num_channels: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: torch.device,
            text_prompt: str,
            short_threshold: Optional[float],
            long_threshold: Optional[float],
            noise_fusion_sigma: Optional[float],
            short_score_fusion_threshold: Optional[float],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        print(f"chunk 0 : Generating {self.beam_size * self.top_k} candidates ...")
        for i in range(self.beam_size * self.top_k):
            cand_seed = self.seed + block_idx * 1000 + i
            candidate = self.generate_candidate(
                noise_shape=(1, current_num_frames, num_channels, height, width),
                conditional_dict=conditional_dict,
                beam=beam,
                block_idx=block_idx,
                current_start_frame=current_start_frame,
                candidate_seed=cand_seed,
                text_prompt=text_prompt,
                output_folder=result_dir,
                dtype=dtype,
                device=device,
                short_threshold=short_threshold,
                long_threshold=long_threshold,
                noise_fusion_sigma=noise_fusion_sigma,
                short_score_fusion_threshold=short_score_fusion_threshold,
            )
            candidates.append(candidate)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            if len(candidates) > self.top_k:
                pruned_cand = candidates.pop()
                self._free_candidate_resources(pruned_cand)
                torch.cuda.empty_cache()

        return candidates

    def _generate_followup_candidates(
            self,
            conditional_dict: Any,
            beams: List[Dict[str, Any]],
            result_dir: str,
            current_start_frame: int,
            block_idx: int,
            current_num_frames: int,
            num_channels: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: torch.device,
            text_prompt: str,
            short_threshold: Optional[float],
            long_threshold: Optional[float],
            noise_fusion_sigma: Optional[float],
            short_score_fusion_threshold: Optional[float],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for beam_idx, beam in enumerate(beams):
            for i in range(self.beam_size):
                if i == 0:
                    cand_seed = beam["history_seeds"][-1]
                else:
                    cand_seed = self.seed + block_idx * 1000 + beam_idx * 100 + i

                candidate = self.generate_candidate(
                    noise_shape=(1, current_num_frames, num_channels, height, width),
                    conditional_dict=conditional_dict,
                    beam=beam,
                    block_idx=block_idx,
                    current_start_frame=current_start_frame,
                    candidate_seed=cand_seed,
                    text_prompt=text_prompt,
                    output_folder=result_dir,
                    dtype=dtype,
                    device=device,
                    short_threshold=short_threshold,
                    long_threshold=long_threshold,
                    noise_fusion_sigma=noise_fusion_sigma,
                    short_score_fusion_threshold=short_score_fusion_threshold,
                    beam_idx=str(beam_idx),
                )
                candidates.append(candidate)
                candidates.sort(key=lambda x: x["score"], reverse=True)
                if len(candidates) > self.top_k:
                    pruned_cand = candidates.pop()
                    self._free_candidate_resources(pruned_cand)

            self._free_candidate_resources(beam)
            torch.cuda.empty_cache()

        return candidates

    def _free_candidate_resources(self, candidate: Dict[str, Any]) -> None:
        if "kv_cache" in candidate:
            del candidate["kv_cache"]
        if "output" in candidate:
            del candidate["output"]
        if "noise" in candidate:
            del candidate["noise"]

    def _write_final_videos(
            self,
            beams: List[Dict[str, Any]],
            text_prompt: str,
            idx: Optional[int],
            device: torch.device,
    ) -> Tuple[Optional[str], float]:
        output_folder = os.path.join(self.output_folder, f"{idx}", "final_videos")
        os.makedirs(output_folder, exist_ok=True)

        best_score = -float('inf')
        best_video: Optional[str] = None
        for beam_idx, beam in enumerate(beams):
            final_output = beam["output"]
            video = self.vae.decode_to_pixel(final_output.to(device), use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = rearrange(video, 'b t c h w -> b t h w c').cpu()
            video = 255.0 * video
            self.vae.model.clear_cache()

            output_path = os.path.join(output_folder, f'beam_{beam_idx}.mp4')
            write_video(output_path, video[0], fps=DEFAULT_VIDEO_FPS)

            score = self.video_reward_inferencer.reward([output_path], [text_prompt])[0]["Overall"]
            if score > best_score:
                best_score = score
                best_video = output_path

        return best_video, best_score

    def _write_best_video_metadata(self, result_dir: str, best_video: Optional[str], best_score: float, text_prompt: str) -> None:
        filename = os.path.join(result_dir, "best_video.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Best Video: {best_video}, Score:{best_score},prompt:{text_prompt}")

        if best_video:
            dst_path = os.path.join(self.output_folder, f"{result_dir.split(os.sep)[-1]}.mp4")
            shutil.copy2(best_video, dst_path)
            print(f"Best video saved as: {dst_path}")

    def _print_profile_results(
            self,
            init_time: float,
            diffusion_time: float,
            vae_time: float,
            total_time: float,
            block_times: List[float],
    ) -> None:
        print("Profiling results:")
        print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
        print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
        for i, block_time in enumerate(block_times):
            print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
        print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
        print(f"  - Total time: {total_time:.2f} ms")

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                # kv_cache_size = self.local_attn_size * self.frame_seq_length
                kv_cache_size = (self.local_attn_size + self.max_sink_size - self.sink_size) * self.frame_seq_length
                kv_policy = f"int->local+sink, local_size={self.local_attn_size}, sink_size={self.sink_size}, max_sink_size={self.max_sink_size}"
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760
            
            print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {self.num_output_frames})")

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "current_sink_size" : torch.tensor([self.sink_size], dtype=torch.long, device=device),
                "current_kv_cache_size" : torch.tensor([self.local_attn_size * self.frame_seq_length], dtype=torch.long, device=device),
                "append_sink_start_frame" : torch.tensor([], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass