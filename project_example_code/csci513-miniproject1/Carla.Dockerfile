FROM carlasim/carla:0.9.15 AS carla

# ---- system setup as root ----
USER root
# Make apt robust & small
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl wget libomp5 xdg-user-dirs \
 && rm -rf /var/lib/apt/lists/*

# Clean up old NVIDIA lists if present (don't fail if absent)
RUN apt-key del 7fa2af80 || true \
 && rm -f /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/cuda.list || true

# (Optional) Install CUDA repo keyring if you genuinely need it
# RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
#  && dpkg -i cuda-keyring_1.0-1_all.deb \
#  && rm -f cuda-keyring_1.0-1_all.deb

# Where we’ll place extra assets
ENV CARLA_ROOT=/opt/carla
RUN install -d -m 0775 "$CARLA_ROOT"

# --- OPTIONAL: you already start FROM carlasim/carla:0.9.15, so the base CARLA is included.
# If you truly need to overwrite it with the tarball, uncomment the block below.
# RUN curl -fL "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz" \
#       -o /tmp/CARLA_0.9.15.tar.gz \
#  && tar -xvzf /tmp/CARLA_0.9.15.tar.gz -C "$CARLA_ROOT" \
#  && rm -f /tmp/CARLA_0.9.15.tar.gz

# Additional Maps for 0.9.15
RUN curl -fL "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz" \
      -o /tmp/AdditionalMaps_0.9.15.tar.gz \
 && tar -xvzf /tmp/AdditionalMaps_0.9.15.tar.gz -C "$CARLA_ROOT" \
 && rm -f /tmp/AdditionalMaps_0.9.15.tar.gz

# Give the non-root user access to the installed content
RUN chown -R carla:carla "$CARLA_ROOT"

# ---- runtime as carla ----
USER carla
WORKDIR /home/carla

# Headless rendering quirk
ENV SDL_VIDEODRIVER=""

# Use JSON-form CMD (better signal handling)
CMD ["/bin/bash","-lc","unset SDL_VIDEODRIVER; ./CarlaUE4.sh -vulkan -RenderOffScreen -nosound"]
