# imdiff

A program for visualizing stereo matching costs

## User Information

### About

This tool provides a visualization of pixel-wise matching costs between two images, for use in stereo vision research. The user can apply transformations to one of the two images in order to line up corresponding objects in the scene. The tool performs best on a rectified stereo pair. It is a native application written in c++ using OpenCV. 

The user can select between several methods of computing the matching cost between two images: 
`color difference`
`ICPR`
`NCC`

### Usage

#### Key Bindings

| Key           | Effect                                   |
|---------------|------------------------------------------|
|drag           |change horizontal offset                  |
|shift-drag     |fine control                              |
|control-drag   |allow vertical offset too                 |
|right-drag     |change disp x and y gradient              |
|shift-right-drag |fine control                              |
|arrows         |change offset by stepsize                 |
|C, V �         |change step size                          |
|O, P��         |change disp x gradient                    |
|[, ]��         |change disp y gradient                    |
|Space          |reset offset                              |
|A, S           |show (blink) orig images                  |
|D              |show diff                                 |
|W              |toggle GT warped image 2                  |
|+              |show confidence measure                   |
|=              |warp by selected plane (default=0)        |
|/              |toggle bounding box for plane             |
|<, >           |cycle through planes                      |
|Y, U           |confidence contrast                       |
|1              |change mode to color diff                 |
|2              |change mode to NCC                        |
|3              |change mode to ICPR cost                  |
|4              |change mode to Bleyer cost                |
|B              |toggle clipping at 0 (modes 2-4)          |
|Z, X           |change diff contrast (mode 1)             |
|E, R           |change NCC epsilon (mode 2)               |
|N, M           |change NCC window size (mode 2)           |
|F, G           |change aggregation window size (modes 2-4)|
|JKLI           |move cropped region (large imgs only)     |
|H , ?          |help                                      |
|(-)            |close current window                      |
|Esc, Q         |quit                                      |

#### Command Line

To run the application from the command line:

usage: ./imdiff im1 im2 [planes=, gt=, occmask=, decimate=, offx=, offy=]

Note: The command line parser does not expect spaces before/after the equals signs. 

#### Dependencies

This program makes use of a pfm reader, which is part of the distribution.
The pfm reader can also be downloaded separately from github via

`git clone https://github.com/dscharstein/pfmLib.git`

 

