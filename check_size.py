from PIL import Image

img = Image.open("data/live/D-3.jpeg")

# Resize image
img = img.resize((224, 224))

# Save resized image
img.save("data/live/D-3_resized.jpeg")

print(img.size)