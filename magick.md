Helpful Magick commands for creating 2DTextureArray

## Texture atlas creator (magick)

Pack ambient occlusion, roughness and displacement into one file.

```bash
magick \
  {AMBIENT_OCCLUSION_FILE} -separate -channel R -combine \
  {ROUGHNESS_FILE} -separate -channel G -combine \
  {DISPLACEMENT_FILE} -separate -channel B -combine \
  -colorspace sRGB \
  {OUTPUT_FILE}
```

Example for ambientcg textures

magick \
 ambient-occlusion.jpg -separate -channel R -combine \
 roughness.jpg -separate -channel G -combine \
 displacement.jpg -separate -channel B -combine \
 -colorspace sRGB \
 aord-packed.jpg

Pack all into texture atlas

```bash
magick \
  \( {tex_1}/albedo.jpg {tex_1}/normal.jpg {tex_1}/aord-packed.jpg -append \) \
  \( {tex_2}/albedo.jpg {tex_2}/normal.jpg {tex_2}/aord-packed.jpg -append \) \
  \ ...
  +append \
  texture_array.png
```

magick \
 \( grass/albedo.jpg grass/normal.jpg grass/aord-packed.jpg -append \) \
 \( dirt/albedo.jpg dirt/normal.jpg dirt/aord-packed.jpg -append \) \
 \( stone/albedo.jpg stone/normal.jpg stone/aord-packed.jpg -append \) \
 \( snow/albedo.jpg snow/normal.jpg snow/aord-packed.jpg -append \) \
 +append \
 texture_array.png
