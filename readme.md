# README

## XXX

we recommend that you choose a language with a library that facilitates image-processing workflow like plantCV
Different methods of direct extraction of characteristics from an image of a leaf need
to be implemented. Once again, you must display at least 6 image transformations.

For exemple:
    $> ./Transformation.[extension] image.jpg

If your program is given a direct path to an image, it must display your set of image
transformations. However, if it is given a source path to a directory filled with multiple
images, it must then save all the image transformations in the specified destination directory.

For exemple:
    $> ./Transformation.[extension] -src Apple/apple_healthy/ -dst dst_directory -mask

Think to make your own usage to facilitate the choice of the
arguments with ./Transformation.[extension] -h

To allow a fast evaluation, it goes without saying that you will
create your own data set from the images by following all these steps
beforehand. Beware, the evaluation will assess whether your program
is working well on small data sets.

### Commandes

    - pyinstaller --onefile Transformation.py
