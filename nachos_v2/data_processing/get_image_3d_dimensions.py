from PIL import Image

def get_image_3d_dimensions(image_path):
    # Ouvrir l'image multi-pages
    img = Image.open(image_path)

    # Lire les dimensions de la première page
    width, height = img.size

    # Compter le nombre de pages (profondeur)
    depth = 0
    while True:
        try:
            img.seek(img.tell() + 1)  # Passer à la page suivante
            depth += 1
        except EOFError:
            break

    return width, height, depth

# Exemple d'utilisation
image_path = '/ssd/sly/CW_treated_tiff/Kidney_02/k2_cortex/k2_cortex_0008_1_Mode3D.tiff'  # Remplacez par le chemin de votre image
width, height, depth = get_image_3d_dimensions(image_path)

print(f'Width: {width}, Height: {height}, Depth: {depth}')
