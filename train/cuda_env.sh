# Environment every GPU job script on Isambard-AI needs. Sourced, not executed:
#   source ./cuda_env.sh
#
# CUDA 13.0 FORWARD-COMPAT LIBS - ONLY WHEN THEY ARE NEWER THAN THE NODE DRIVER.
# Until early October 2026 the nodes ran driver 565.57.01 (CUDA 12.7), and a cu130
# torch build could only see the GPU with the compat libcuda ahead of the system
# one; without it torch fell back to CPU in silence. The nodes then moved to
# 580.173.02, which supports CUDA 13.0 natively, and the compat libcuda
# (580.65.06) became OLDER than the kernel driver - which is CUDA error 803. So
# prepend the compat path only when it is newer than the system libcuda. The
# same block is inline in train_samsum.sbatch and evaluate.sbatch.
CUDA_COMPAT=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat
sys_libcuda=$(ls /usr/lib64/libcuda.so.*.* 2>/dev/null | sed 's/.*libcuda\.so\.//' | sort -V | tail -1)
compat_libcuda=$(ls "$CUDA_COMPAT"/libcuda.so.*.* 2>/dev/null | sed 's/.*libcuda\.so\.//' | sort -V | tail -1)
if [[ -n $compat_libcuda && $compat_libcuda != "$sys_libcuda" \
      && $(printf '%s\n%s\n' "$sys_libcuda" "$compat_libcuda" | sort -V | tail -1) == "$compat_libcuda" ]]; then
  export LD_LIBRARY_PATH=$CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  echo "cuda: driver $sys_libcuda older than compat $compat_libcuda - using compat libs"
else
  echo "cuda: driver ${sys_libcuda:-unknown} not older than compat ${compat_libcuda:-none} - using system libcuda"
fi

# Never re-resolve the environment inside a job: array tasks share one venv and
# concurrent syncs race on it. Sync once on a login node, with --inexact so a
# sync can only add packages - removing a leftover cu12 wheel once deleted files
# the cu13 cuDNN wheel shares with it, and broke torch in every venv.
export UV_NO_SYNC=1

# HF_HUB_CACHE, NOT HF_HOME: HF_HOME also moves the auth token.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECTDIR/$USER/mm/hf-cache}

# Air-gapped compute nodes: telemetry retries against the network regardless.
export DO_NOT_TRACK=1
export AXOLOTL_DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
