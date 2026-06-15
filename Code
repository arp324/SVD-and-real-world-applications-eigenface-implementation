from PIL import Image
import numpy
import glob
import matplotlib.pyplot as plt

paths = glob.glob("faces/*.jpg")

# Load images as grayscale, normalized matrices
# The target size makes the images smaller so SVD is faster
# Created a 3D matrix of shape (num_images, height, width)
target_size = (400, 400)
images = numpy.array([numpy.array(Image.open(n).convert("L").resize(target_size)) / 255.0 for n in paths])

# Calculating the mean face by finding the mean intensity value for each pixel across all images
mean_face = numpy.mean(images, axis=0)
height, width = mean_face.shape

# Flatten images into data matrix
# Reshaping the 3D matrix of images into a 2D matrix where each row corresponds to a flattened image
# The -1 in reshape refers to the reduction in dimension
flattened_data = images.reshape(len(images), -1)

# Center data
# Subtracts the mean face from each flattened image
centered_data = flattened_data - mean_face.flatten()

# Efficient SVD via small Gram matrix
# Gram matrix has dimension (number of images)^2 instead of (number of pixels)^2
gram = centered_data @ centered_data.T
U, S_sq, _ = numpy.linalg.svd(gram)
eigenfaces = (centered_data.T @ U).T
eigenfaces /= numpy.linalg.norm(eigenfaces, axis=1, keepdims=True)

# Singular values for explained variance (sqrt of gram eigenvalues)
# Eigenvalues of gram equal square of singular values of centered_data
S = numpy.sqrt(numpy.maximum(S_sq, 0))

# Choose k by cumulative explained variance (retain 95%)
explained = numpy.cumsum(S**2) / numpy.sum(S**2)
k = int(numpy.searchsorted(explained, 0.95)) + 1
k = min(k, len(paths))
print(f"k={k} components explain 95% of variance")

# Select top k eigenfaces
eigenfaces = eigenfaces[:k]
eigenface_images = eigenfaces.reshape(k, height, width)

# Project training faces into face space
face_weights = centered_data @ eigenfaces.T

# Create distance threshold via reconstruction error
reconstructed = face_weights @ eigenfaces
errors = numpy.linalg.norm(centered_data - reconstructed, axis=1)
threshold = numpy.mean(errors) + 2 * numpy.std(errors)

# Create residual threshold via reconstruction error
train_residuals = [
    numpy.linalg.norm(centered_data[i] - face_weights[i] @ eigenfaces)
    for i in range(len(face_weights))
]
residual_threshold = numpy.percentile(train_residuals, 95)
print(f"Auto-calibrated residual threshold: {residual_threshold:.4f}")

# Print key shapes
print("images:", images.shape)
print("mean_face:", mean_face.shape)
print("flattened_data:", flattened_data.shape)
print("centered_data:", centered_data.shape)
print("eigenfaces:", eigenfaces.shape)
print("face_weights:", face_weights.shape)

# Visualise top 10 eigenfaces
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, ef in zip(axes.flat, eigenface_images[:10]):
    ef_norm = (ef - ef.min()) / (ef.max() - ef.min())
    ax.imshow(ef_norm, cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig("eigenfaces.png", dpi=150)
plt.close()

# New face to recognize
new_path = "new_face.jpg"

# Resizing the new image to be the same as the training images
new_img = Image.open(new_path).convert("L").resize(target_size)
new_arr = numpy.array(new_img) / 255.0

# Flatten and center using the same mean face
new_vector = new_arr.flatten()
new_centered = new_vector - mean_face.flatten()

# Project into face space
new_weights = new_centered @ eigenfaces.T

# Face-space residual check (is this a face?)
reconstruction = new_weights @ eigenfaces
residual = numpy.linalg.norm(new_centered - reconstruction)

# Compare to stored face weights
distances = numpy.linalg.norm(face_weights - new_weights, axis=1)
best_match_index = numpy.argmin(distances)
best_match_path = paths[best_match_index]
best_distance = distances[best_match_index]

# Confidence score (0-1, is only meaningful when a match is found)
confidence = float(numpy.clip(1.0 - (best_distance / threshold), 0, 1))

print("Closest match:", best_match_path)
print("Distance:", best_distance)
print(f"Confidence: {confidence:.2%}")

if residual > residual_threshold:
    print("Result: input does not appear to be a face")
elif best_distance < threshold:
    print(f"Result: match - {best_match_path} ({confidence:.2%} confidence)")
else:
    print("Result: unknown face")

# Printing mean face
plt.imshow(mean_face, cmap="gray")
plt.axis("off")
plt.savefig("mean_face.png", dpi=150)
plt.close()

# Closest matched face
matched_img = Image.open(best_match_path).convert("L").resize(target_size)
matched_arr = numpy.array(matched_img) / 255.0

# Printing test face vs closest face side by side
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

axes[0].imshow(new_arr, cmap="gray")
axes[0].set_title("Test Face")
axes[0].axis("off")
axes[1].imshow(matched_arr, cmap="gray")
axes[1].set_title("Closest Match")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("test_vs_match.png", dpi=150)
plt.close()

# Printing grayscale compression with different K values
img = Image.open("new_face.jpg").convert("L").resize(target_size)
A = numpy.array(img) / 255.0
k_values = [1, 2, 5, 10, 20, 50, 100]

# Compute full SVD once, reuse for all k values
U, S, Vt = numpy.linalg.svd(A, full_matrices=False)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()
axes[0].imshow(A, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

for ax, k in zip(axes[1:], k_values):
    k = min(k, len(S))
    A_k = (U[:, :k] * S[:k]) @ Vt[:k, :]
    ax.imshow(numpy.clip(A_k, 0, 1), cmap="gray")
    ax.set_title(f"k={k}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("svd_grayscale_compression.png", dpi=150)
plt.show()
plt.close()
