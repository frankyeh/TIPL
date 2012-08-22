#ifndef TIFF_TAG_HPP
#define TIFF_TAG_HPP

namespace TiffTag{
// Tag value
const unsigned short PhotometricInterpretation = 262;
/*
 0 = WhiteIsZero. For bilevel and grayscale images: 0 is imaged as white.
1 = BlackIsZero. For bilevel and grayscale images: 0 is imaged as black.
2 = RGB. RGB value of (0,0,0) represents black, and (255,255,255) represents white, assuming 8-bit components. The components are stored in the indicated order: first Red, then Green, then Blue.
3 = Palette color. In this model, a color is described with a single component. The value of the component is used as an index into the red, green and blue curves in the ColorMap field to retrieve an RGB triplet that defines the color. When PhotometricInterpretation=3 is used, ColorMap must be present and SamplesPerPixel must be 1.
4 = Transparency Mask. This means that the image is used to define an irregularly shaped region of another image in the same TIFF file. SamplesPerPixel and BitsPerSample must be 1. PackBits compression is recommended. The 1-bits define the interior of the region; the 0-bits define the exterior of the region.

These values are considered an extension:

5 = Seperated, usually CMYK.
6 = YCbCr
8 = CIE L*a*b* (see also specification supplements 1 and 2)
9 = CIE L*a*b*, alternate encoding also known as ICC L*a*b* (see also specification supplements 1 and 2)

The TIFF-F specification (RFC 2301) defines:

10 = CIE L*a*b*, alternate encoding also known as ITU L*a*b*, defined in ITU-T Rec. T.42, used in the TIFF-F and TIFF-FX standard (RFC 2301). The Decode tag, if present, holds information about this particular CIE L*a*b* encoding. 
*/
const unsigned short Compression = 259;
/*
 1 = No compression
2 = CCITT modified Huffman RLE
32773 = PackBits compression, aka Macintosh RLE

Additionally, the specification defines these values as part of the TIFF extensions:

3 = CCITT Group 3 fax encoding
4 = CCITT Group 4 fax encoding
5 = LZW
6 = JPEG ('old-style' JPEG, later overriden in Technote2)

Technote2 overrides old-style JPEG compression, and defines:

7 = JPEG ('new-style' JPEG)

Adobe later added the deflate compression scheme:

8 = Deflate ('Adobe-style')

The TIFF-F specification (RFC 2301) defines:

9 = Defined by TIFF-F and TIFF-FX standard (RFC 2301) as ITU-T Rec. T.82 coding, using ITU-T Rec. T.85 (which boils down to JBIG on black and white).
10 = Defined by TIFF-F and TIFF-FX standard (RFC 2301) as ITU-T Rec. T.82 coding, using ITU-T Rec. T.43 (which boils down to JBIG on color). 
*/
const unsigned short ImageLength = 257;
const unsigned short ImageWidth = 256;
const unsigned short ResolutionUnit = 296;
/*
No absolute unit of measurement. Used for images that may have a non-square
aspect ratio but no meaningful absolute dimensions.
2 = Inch.
3 = Centimeter.
Default = 2 (inch).
*/
const unsigned short XResolutionUnit = 282;
const unsigned short YResolutionUnit = 283;
const unsigned short RowsPerStrip = 278;
/*
The number of rows in each strip (except possibly the last strip.)
For example, if ImageLength is 24, and RowsPerStrip is 10, then there are 3
strips, with 10 rows in the first strip, 10 rows in the second strip, and 4 rows in the
third strip. (The data in the last strip is not padded with 6 extra rows of dummy
data.)*/
const unsigned short StripOffsets = 273;
const unsigned short StripByteCounts = 279;

/*For each strip, the number of bytes in that strip after any compression.
Putting it all together (along with a couple of less-important fields that are discussed
later), */
const unsigned short BitsPerSample = 258;
/*
The number of bits per component.
Allowable values for Baseline TIFF grayscale images are 4 and 8, allowing either
16 or 256 distinct shades of gray.
*/

const unsigned short ColorMap = 320;
//PhotometricInterpretation = 3 (Palette Color).
/*
N = 3 * (2**BitsPerSample)
This field defines a Red-Green-Blue color map (often called a lookup table) for
palette color images. In a palette-color image, a pixel value is used to index into an
RGB-lookup table. For example, a palette-color pixel having a value of 0 would
be displayed according to the 0th Red, Green, Blue triplet.
In a TIFF ColorMap, all the Red values come first, followed by the Green values,
then the Blue values. In the ColorMap, black is represented by 0,0,0 and white is
represented by 65535, 65535, 65535.
*/


const unsigned short SamplesPerPixel = 277;
/*
The number of components per pixel. This number is 3 for RGB images, unless
extra samples are present. See the ExtraSamples field for further information.

BitsPerSample = 8,8,8. Each component is 8 bits deep in a Baseline TIFF RGB image.
PhotometricInterpretation = 2 (RGB).
There is no ColorMap.
*/

}

#endif