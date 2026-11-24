import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# Load original logo
img = Image.open('also_alien_ship.png').convert("RGBA")
data = np.array(img).astype(float)

# Also Alien background color
bg = np.array([25.0, 25.0, 25.0])
rgb = data[:, :, :3]

# Calculate exact color distance to isolate the logo
dist = np.linalg.norm(rgb - bg, axis=2)
min_d, max_d = 2.0, 40.0
alpha = np.clip((dist - min_d) / (max_d - min_d), 0.0, 1.0)

# Un-premultiply the background out of the anti-aliased edge pixels
alpha_safe = np.where(alpha < 0.05, 1.0, alpha)
recovered_rgb = (rgb - bg) / alpha_safe[:, :, None] + bg
recovered_rgb = np.clip(recovered_rgb, 0, 255)

# Flood-fill transparent areas with nearest solid color to prevent dark fringes during upscale
mask = alpha > 0
dist_bg, indices = distance_transform_edt(~mask, return_indices=True)
recovered_rgb[~mask] = recovered_rgb[indices[0][~mask], indices[1][~mask]]

# Reassemble cleanly
out_data = np.zeros_like(data)
out_data[:, :, :3] = recovered_rgb
out_data[:, :, 3] = alpha * 255.0
out_img = Image.fromarray(out_data.astype(np.uint8))

# 8x Upscale for high-res output
final_size = (img.width * 6, img.height * 6)
out_img = out_img.resize(final_size, Image.Resampling.LANCZOS)

out_img.save('Also_Alien_Logo_Transparent_HQ.png')
print("High-resolution transparent PNG saved successfully.")
