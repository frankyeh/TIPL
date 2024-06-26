//---------------------------------------------------------------------------
#ifndef dicom_headerH
#define dicom_headerH
// Copyright Fang-Cheng Yeh 2010
// Distributed under the BSD License
//
/*
Copyright (c) 2010, Fang-Cheng Yeh
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iomanip>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <memory>
#include <locale>
#include "../numerical/basic_op.hpp"
//---------------------------------------------------------------------------
namespace tipl
{
namespace io
{


//---------------------------------------------------------------------------
// decode_1_2_840_10008_1_2_4_70
// modified from https://www.mccauslandcenter.sc.edu/crnl/tools/jpeg-formats
//---------------------------------------------------------------------------
struct HufTables {
    uint8_t SSSSszRA[18];
    uint8_t LookUpRA[256];
    int DHTliRA[32];
    int DHTstartRA[32];
    int HufSz[32];
    int HufCode[32];
    int HufVal[32];
    int MaxHufSi;
    int MaxHufVal;
}; //end HufTables()
const int bitMask[16] = {0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767 };

inline int dcm_decode_pixel(unsigned char *buf_ptr, long *buf_pos, int *bit_pos, struct HufTables l)
{
    int lByte = (buf_ptr[*buf_pos] << *bit_pos) + (buf_ptr[*buf_pos+1] >> (8- *bit_pos));
    lByte = lByte & 255;
    int lHufValSSSS = l.LookUpRA[lByte];
    if (lHufValSSSS < 255) {
        *bit_pos = l.SSSSszRA[lHufValSSSS] + *bit_pos;
        *buf_pos = *buf_pos + (*bit_pos >> 3);
        *bit_pos = *bit_pos & 7;
    } else { //full SSSS is not in the first 8-bits
        int lInput = lByte;
        int lInputBits = 8;
        (*buf_pos)++; // forward 8 bits = precisely 1 byte
        do {
            lInputBits++;
            lInput = (lInput << 1);
            //Read the next single bit
            {
                lInput += (buf_ptr[*buf_pos] >> (7 - *bit_pos)) & 1;
                (*bit_pos)++;
                if (*bit_pos == 8) {
                    (*buf_pos)++;
                    *bit_pos = 0;
                }
            }
            if (l.DHTliRA[lInputBits] != 0) { //if any entries with this length
                for (int lI = l.DHTstartRA[lInputBits]; lI <= (l.DHTstartRA[lInputBits]+l.DHTliRA[lInputBits]-1); lI++) {
                    if (lInput == l.HufCode[lI])
                        lHufValSSSS = l.HufVal[lI];
                } //check each code
            } //if any entries with this length
            if ((lInputBits >= l.MaxHufSi) && (lHufValSSSS > 254)) //exhausted options CR: added rev13
                lHufValSSSS = l.MaxHufVal;
        } while (!(lHufValSSSS < 255)); // found;
    } //answer in first 8 bits
    //The HufVal is referred to as the SSSS in the Codec, so it is called 'lHufValSSSS'
    if (lHufValSSSS == 0) //NO CHANGE
      return 0;
    else if (lHufValSSSS == 16) { //ALL CHANGE 16 bit difference: Codec H.1.2.2 "No extra bits are appended after SSSS = 16 is encoded." Osiris Cornell Libraries fail here
        return 32768; //see H.1.2.2 and table H.2 of ISO 10918-1
    }
    //to get here - there is a 1..15 bit difference
    int lDiff = buf_ptr[*buf_pos];
    lDiff = (lDiff << 8) + buf_ptr[(*buf_pos)+1];
    lDiff = (lDiff << 8) + buf_ptr[(*buf_pos)+2];
    lDiff = (lDiff >> (24 - *bit_pos -lHufValSSSS)) & bitMask[lHufValSSSS]; //bit_pos is incremented from 1, so -1
    *bit_pos = *bit_pos + lHufValSSSS;
    if (*bit_pos > 7) {
            *buf_pos = *buf_pos + (*bit_pos >> 3); // div 8
            *bit_pos = *bit_pos & 7; //mod 8
    }
    if (lDiff <= bitMask[lHufValSSSS-1] )//add
        lDiff = lDiff - bitMask[lHufValSSSS];
    return lDiff;
} //end dcm_decode_pixel()

inline uint16_t dcm_read_word(unsigned char *buf_ptr, long *buf_pos) {
    uint16_t ret = (uint16_t(buf_ptr[*buf_pos]) << 8) + uint16_t(buf_ptr[*buf_pos+1]);
    (*buf_pos) += 2;
    return ret;
} //end dcm_read_word()

inline bool decode_1_2_840_10008_1_2_4_70(unsigned char *buf_ptr, long buf_size, std::vector<unsigned char>& buf,
                                       int *dimX, int *dimY, int *bits, int *frames)
{
    unsigned char *lImgRA8 = nullptr;
    if ((buf_ptr[0] != 0xFF) || (buf_ptr[1] != 0xD8) || (buf_ptr[2] != 0xFF))
        return false;
    //next: read header
    long buf_pos = 2; //Skip initial 0xFFD8, begin with third byte
    unsigned char btS1, SOSss(0), SOSahal, SOSpttrans(0), btMarkerType, SOSns = 0x00; //tag
    uint8_t SOFnf(0), SOFprecision(0);
    uint16_t SOFydim(0), SOFxdim(0); //, lRestartSegmentSz;
    int lnHufTables = 0;
    const int kmaxFrames = 4;
    struct HufTables l[kmaxFrames+1];
    do { //read each marker in the header
        do {
            btS1 = buf_ptr[buf_pos++];
            if (btS1 != 0xFF) {
                printf("JPEG header tag must begin with 0xFF\n");
                return false;
            }
            btMarkerType =  buf_ptr[buf_pos++];
            if ((btMarkerType == 0x01) || (btMarkerType == 0xFF) || ((btMarkerType >= 0xD0) && (btMarkerType <= 0xD7) ) )
                btMarkerType = 0;//only process segments with length fields

        } while ((buf_pos < buf_size) && (btMarkerType == 0));
        uint16_t lSegmentLength = dcm_read_word (buf_ptr, &buf_pos); //read marker length
        long lSegmentEnd = buf_pos+(lSegmentLength - 2);
        if (lSegmentEnd > buf_size)  {
            return false;
        }
        if ( ((btMarkerType >= 0xC0) && (btMarkerType <= 0xC3)) || ((btMarkerType >= 0xC5) && (btMarkerType <= 0xCB)) || ((btMarkerType >= 0xCD) && (btMarkerType <= 0xCF)) )  {
            //if Start-Of-Frame (SOF) marker
            SOFprecision = buf_ptr[buf_pos++];
            SOFydim = dcm_read_word(buf_ptr, &buf_pos);
            SOFxdim = dcm_read_word(buf_ptr, &buf_pos);
            SOFnf = buf_ptr[buf_pos++];
            buf_pos = (lSegmentEnd);
            if (btMarkerType != 0xC3) { //lImgTypeC3 = true;
                printf("This JPEG decoder can only decompress lossless JPEG ITU-T81 images (SoF must be 0XC3, not %#02X)\n",btMarkerType );
                return false;
            }
            if ( (SOFprecision < 1) || (SOFprecision > 16) || (SOFnf < 1) || (SOFnf == 2) || (SOFnf > 3)
                || ((SOFnf == 3) &&  (SOFprecision > 8))   ) {
                printf("Scalar data must be 1..16 bit, RGB data must be 8-bit (%d-bit, %d frames)\n", SOFprecision, SOFnf);
                return false;
            }
        } else if (btMarkerType == 0xC4) {//if SOF marker else if define-Huffman-tables marker (DHT)
            int lFrameCount = 1;
            do {
                uint8_t DHTnLi = buf_ptr[buf_pos++]; //we read but ignore DHTtcth.
                DHTnLi = 0;
                for (int lInc = 1; lInc <= 16; lInc++) {
                    l[lFrameCount].DHTliRA[lInc] = buf_ptr[buf_pos++];
                    DHTnLi = DHTnLi +  l[lFrameCount].DHTliRA[lInc];
                    if (l[lFrameCount].DHTliRA[lInc] != 0) l[lFrameCount].MaxHufSi = lInc;
                }
                if (DHTnLi > 17) {
                    printf("Huffman table corrupted.\n");
                    return false;
                }
                int lIncY = 0; //frequency
                for (int lInc = 0; lInc <= 31; lInc++) {//lInc := 0 to 31 do begin
                    l[lFrameCount].HufVal[lInc] = -1;
                    l[lFrameCount].HufSz[lInc] = -1;
                    l[lFrameCount].HufCode[lInc] = -1;
                }
                for (int lInc = 1; lInc <= 16; lInc++) {//set the huffman size values
                    if (l[lFrameCount].DHTliRA[lInc] > 0) {
                        l[lFrameCount].DHTstartRA[lInc] = lIncY+1;
                        for (int lIncX = 1; lIncX <= l[lFrameCount].DHTliRA[lInc]; lIncX++) {
                            lIncY++;
                            btS1 = buf_ptr[buf_pos++];
                            l[lFrameCount].HufVal[lIncY] = btS1;
                            l[lFrameCount].MaxHufVal = btS1;
                            if (btS1 <= 16)
                                l[lFrameCount].HufSz[lIncY] = lInc;
                            else {
                                printf("Huffman size array corrupted.\n");
                                return false;
                            }
                        }
                    }
                } //set huffman size values
                int K = 1;
                int Code = 0;
                int Si = l[lFrameCount].HufSz[K];
                do {
                    while (Si == l[lFrameCount].HufSz[K]) {
                        l[lFrameCount].HufCode[K] = Code;
                        Code = Code + 1;
                        K++;
                    }
                    if (K <= DHTnLi) {
                        while (l[lFrameCount].HufSz[K] > Si) {
                            Code = Code << 1; //Shl!!!
                            Si = Si + 1;
                        }//while Si
                    }//K <= 17

                } while (K <= DHTnLi);
                lFrameCount++;
            } while ((lSegmentEnd-buf_pos) >= 18);
            lnHufTables = lFrameCount - 1;
            buf_pos = (lSegmentEnd);
        } else if (btMarkerType == 0xDD) {  //if DHT marker else if Define restart interval (DRI) marker
            printf("This image uses Restart Segments - please contact Chris Rorden to add support for this format.\n");
            return false;
            //lRestartSegmentSz = dcm_read_word(buf_ptr, &buf_pos, buf_size);
            //buf_pos = lSegmentEnd;
        } else if (btMarkerType == 0xDA) {  //if DRI marker else if read Start of Scan (SOS) marker
            SOSns = buf_ptr[buf_pos++];
            //if Ns = 1 then NOT interleaved, else interleaved: see B.2.3
            if (SOSns > 0) {
                for (int lInc = 1; lInc <= SOSns; lInc++) {
                    btS1 = buf_ptr[buf_pos++]; //component identifier 1=Y,2=Cb,3=Cr,4=I,5=Q
                    buf_pos++; //horizontal and vertical sampling factors
                }
            }
            SOSss = buf_ptr[buf_pos++]; //predictor selection B.3
            buf_pos++;
            SOSahal = buf_ptr[buf_pos++]; //lower 4bits= pointtransform
            SOSpttrans = SOSahal & 16;
            buf_pos = (lSegmentEnd);
        } else  //if SOS marker else skip marker
            buf_pos = (lSegmentEnd);
    } while ((buf_pos < buf_size) && (btMarkerType != 0xDA)); //0xDA=Start of scan: loop for reading header
    //NEXT: Huffman decoding
    if (lnHufTables < 1)
        return false;
    // Decoding error: no Huffman tables
    //NEXT: unpad data - delete byte that follows $FF
    long lIncI = buf_pos; //input position
    long lIncO = buf_pos; //output position
    do {
        buf_ptr[lIncO] = buf_ptr[lIncI];
        if (buf_ptr[lIncI] == 255) {
            if (buf_ptr[lIncI+1] == 0)
                lIncI = lIncI+1;
            else if (buf_ptr[lIncI+1] == 0xD9)
                lIncO = -666; //end of padding
        }
        lIncI++;
        lIncO++;
    } while (lIncO > 0);
    //NEXT: some RGB images use only a single Huffman table for all 3 colour planes. In this case, replicate the correct values
    //NEXT: prepare lookup table

    for (int lFrameCount = 1; lFrameCount <= lnHufTables; lFrameCount ++) {
        for (int lInc = 0; lInc <= 17; lInc ++)
            l[lFrameCount].SSSSszRA[lInc] = 123; //Impossible value for SSSS, suggests 8-bits can not describe answer
        for (int lInc = 0; lInc <= 255; lInc ++)
            l[lFrameCount].LookUpRA[lInc] = 255; //Impossible value for SSSS, suggests 8-bits can not describe answer
    }
    //NEXT: fill lookuptable


    for (int lFrameCount = 1; lFrameCount <= lnHufTables; lFrameCount ++) {
        int lIncY = 0;
        for (int lSz = 1; lSz <= 8; lSz ++) { //set the huffman lookup table for keys with lengths <=8
            if (l[lFrameCount].DHTliRA[lSz]> 0) {
                for (int lIncX = 1; lIncX <= l[lFrameCount].DHTliRA[lSz]; lIncX ++) {
                    lIncY++;
                    int lHufVal = l[lFrameCount].HufVal[lIncY]; //SSSS
                    l[lFrameCount].SSSSszRA[lHufVal] = lSz;
                    int k = (l[lFrameCount].HufCode[lIncY] << (8-lSz )) & 255; //K= most sig bits for hufman table
                    if (lSz < 8) { //fill in all possible bits that exceed the huffman table
                        int lInc = bitMask[8-lSz];
                        for (int bit_pos = 0; bit_pos <= lInc; bit_pos++) {
                            l[lFrameCount].LookUpRA[k+bit_pos] = lHufVal;
                        }
                    } else
                        l[lFrameCount].LookUpRA[k] = lHufVal; //SSSS
                    //printf("Frame %d SSSS %d Size %d Code %d SHL %d EmptyBits %ld\n", lFrameCount, lHufRA[lFrameCount][lIncY].HufVal, lHufRA[lFrameCount][lIncY].HufSz,lHufRA[lFrameCount][lIncY].HufCode, k, lInc);
                } //Set SSSS
            } //Length of size lInc > 0
        } //for lInc := 1 to 8
    } //For each frame, e.g. once each for Red/Green/Blue
    //NEXT: some RGB images use only a single Huffman table for all 3 colour planes. In this case, replicate the correct values
    if (lnHufTables < SOFnf) { //use single Hufman table for each frame
        for (int lFrameCount = 2; lFrameCount <= SOFnf; lFrameCount++) {
            l[lFrameCount] = l[1];
        } //for each frame
    } // if lnHufTables < SOFnf
    //NEXT: uncompress data: different loops for different predictors
    int lItems =  SOFxdim*SOFydim*SOFnf;
    // buf_pos++;// <- only for Pascal where array is indexed from 1 not 0 first byte of data
    int bit_pos = 0; //read in a new byte

    //depending on SOSss, we see Table H.1
    int lPredA = 0;
    int lPredB = 0;
    int lPredC = 0;
    if (SOSss == 2) //predictor selection 2: above
        lPredA = SOFxdim-1;
    else if (SOSss == 3) //predictor selection 3: above+left
        lPredA = SOFxdim;
    else if ((SOSss == 4) || (SOSss == 5)) { //these use left, above and above+left WEIGHT LEFT
        lPredA = 0; //Ra left
        lPredB = SOFxdim-1; //Rb directly above
        lPredC = SOFxdim; //Rc UpperLeft:above and to the left
    } else if (SOSss == 6) { //also use left, above and above+left, WEIGHT ABOVE
        lPredB = 0;
        lPredA = SOFxdim-1; //Rb directly above
        lPredC = SOFxdim; //Rc UpperLeft:above and to the left
    }   else
        lPredA = 0; //Ra: directly to left)
    if (SOFprecision > 8) { //start - 16 bit data
        *bits = 16;
        int lPx = -1; //pixel position
        int lPredicted =  1 << (SOFprecision-1-SOSpttrans);
        buf.resize(lItems*2);
        lImgRA8 = buf.data();
        uint16_t *lImgRA16 = (uint16_t*) lImgRA8;
        for (int i = 0; i < lItems; i++)
            lImgRA16[i] = 0; //zero array
        int frame = 1;
        for (int lIncX = 1; lIncX <= SOFxdim; lIncX++) { //for first row - here we ALWAYS use LEFT as predictor
            lPx++; //writenext voxel
            if (lIncX > 1) lPredicted = lImgRA16[lPx-1];
            lImgRA16[lPx] = lPredicted+ dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[frame]);
        }
        for (int lIncY = 2; lIncY <= SOFydim; lIncY++) {//for all subsequent rows
            lPx++; //write next voxel
            lPredicted = lImgRA16[lPx-SOFxdim]; //use ABOVE
            lImgRA16[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[frame]);
            if (SOSss == 4) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA16[lPx-lPredA]+lImgRA16[lPx-lPredB]-lImgRA16[lPx-lPredC];
                    lPx++; //writenext voxel
                    lImgRA16[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[frame]);
                } //for lIncX
            } else if ((SOSss == 5) || (SOSss == 6)) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA16[lPx-lPredA]+ ((lImgRA16[lPx-lPredB]-lImgRA16[lPx-lPredC]) >> 1);
                    lPx++; //writenext voxel
                    lImgRA16[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[frame]);
                } //for lIncX
            } else if (SOSss == 7) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPx++; //writenext voxel
                    lPredicted = (lImgRA16[lPx-1]+lImgRA16[lPx-SOFxdim]) >> 1;
                    lImgRA16[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[frame]);
                } //for lIncX
            } else { //SOSss 1,2,3 read single values
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA16[lPx-lPredA];
                    lPx++; //writenext voxel
                    lImgRA16[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[frame]);
                } //for lIncX
            } // if..else possible predictors
        }//for lIncY
    } else if (SOFnf == 3) { //if 16-bit data; else 8-bit 3 frames
        *bits = 8;
        buf.resize(lItems);
        lImgRA8 = buf.data();

        int lPx[kmaxFrames+1], lPredicted[kmaxFrames+1]; //pixel position
        for (int f = 1; f <= SOFnf; f++) {
            lPx[f] = ((f-1) * (SOFxdim * SOFydim) ) -1;
            lPredicted[f] = 1 << (SOFprecision-1-SOSpttrans);
        }
        for (int i = 0; i < lItems; i++)
            lImgRA8[i] = 255; //zero array
        for (int lIncX = 1; lIncX <= SOFxdim; lIncX++) { //for first row - here we ALWAYS use LEFT as predictor
            for (int f = 1; f <= SOFnf; f++) {
                lPx[f]++; //writenext voxel
                if (lIncX > 1) lPredicted[f] = lImgRA8[lPx[f]-1];
                lImgRA8[lPx[f]] = lPredicted[f] + dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[f]);
            }
        } //first row always predicted by LEFT
        for (int lIncY = 2; lIncY <= SOFydim; lIncY++) {//for all subsequent rows
            for (int f = 1; f <= SOFnf; f++) {
                lPx[f]++; //write next voxel
                lPredicted[f] = lImgRA8[lPx[f]-SOFxdim]; //use ABOVE
                lImgRA8[lPx[f]] = lPredicted[f] + dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[f]);
            }//first column of row always predicted by ABOVE
            if (SOSss == 4) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA8[lPx[f]-lPredA]+lImgRA8[lPx[f]-lPredB]-lImgRA8[lPx[f]-lPredC];
                        lPx[f]++; //writenext voxel
                        lImgRA8[lPx[f]] = lPredicted[f]+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[f]);
                    }
                } //for lIncX
            } else if ((SOSss == 5) || (SOSss == 6)) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA8[lPx[f]-lPredA]+ ((lImgRA8[lPx[f]-lPredB]-lImgRA8[lPx[f]-lPredC]) >> 1);
                        lPx[f]++; //writenext voxel
                        lImgRA8[lPx[f]] = lPredicted[f] + dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[f]);
                    }
                } //for lIncX
            } else if (SOSss == 7) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPx[f]++; //writenext voxel
                        lPredicted[f] = (lImgRA8[lPx[f]-1]+lImgRA8[lPx[f]-SOFxdim]) >> 1;
                        lImgRA8[lPx[f]] = lPredicted[f] + dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[f]);
                    }
                } //for lIncX
            } else { //SOSss 1,2,3 read single values
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA8[lPx[f]-lPredA];
                        lPx[f]++; //writenext voxel
                        lImgRA8[lPx[f]] = lPredicted[f] + dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[f]);
                    }
                } //for lIncX
            } // if..else possible predictors
        }//for lIncY
    }else { //if 8-bit data 3frames; else 8-bit 1 frames
        *bits = 8;
        buf.resize(lItems);
        lImgRA8 = buf.data();
        int lPx = -1; //pixel position
        int lPredicted =  1 << (SOFprecision-1-SOSpttrans);
        for (int i = 0; i < lItems; i++)
            lImgRA8[i] = 0; //zero array
        for (int lIncX = 1; lIncX <= SOFxdim; lIncX++) { //for first row - here we ALWAYS use LEFT as predictor
            lPx++; //writenext voxel
            if (lIncX > 1) lPredicted = lImgRA8[lPx-1];
            int dx = dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[1]);
            lImgRA8[lPx] = lPredicted+dx;
        }
        for (int lIncY = 2; lIncY <= SOFydim; lIncY++) {//for all subsequent rows
            lPx++; //write next voxel
            lPredicted = lImgRA8[lPx-SOFxdim]; //use ABOVE
            lImgRA8[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[1]);
            if (SOSss == 4) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA8[lPx-lPredA]+lImgRA8[lPx-lPredB]-lImgRA8[lPx-lPredC];
                    lPx++; //writenext voxel
                    lImgRA8[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[1]);
                } //for lIncX
            } else if ((SOSss == 5) || (SOSss == 6)) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA8[lPx-lPredA]+ ((lImgRA8[lPx-lPredB]-lImgRA8[lPx-lPredC]) >> 1);
                    lPx++; //writenext voxel
                    lImgRA8[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[1]);
                } //for lIncX
            } else if (SOSss == 7) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPx++; //writenext voxel
                    lPredicted = (lImgRA8[lPx-1]+lImgRA8[lPx-SOFxdim]) >> 1;
                    lImgRA8[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[1]);
                } //for lIncX
            } else { //SOSss 1,2,3 read single values
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA8[lPx-lPredA];
                    lPx++; //writenext voxel
                    lImgRA8[lPx] = lPredicted+dcm_decode_pixel(buf_ptr, &buf_pos, &bit_pos, l[1]);
                } //for lIncX
            } // if..else possible predictors
        }//for lIncY
    } //if 16bit else 8bit
    *dimX = SOFxdim;
    *dimY = SOFydim;
    *frames = SOFnf;
    return lImgRA8;
}
//---------------------------------------------------------------------------

enum transfer_syntax_type {lee,bee,lei};
//---------------------------------------------------------------------------
const char dicom_long_flag[] = "OBOFUNOWSQUT";
//---------------------------------------------------------------------------
class dicom_group_element
{
public:
    union
    {
        char gel[8];
        struct
        {
            unsigned short group;
            unsigned short element;
            union
            {
                unsigned int length;
                struct
                {
                    union
                    {
                        unsigned short vr;
                        struct
                        {
                            char lt0;
                            char lt1;
                        };
                    };
                    union
                    {
                        unsigned short new_length;
                        struct
                        {
                            char lt2;
                            char lt3;
                        };
                    };
                };
            };
        };
    };
    std::vector<unsigned char> data;
    // for VR=SQ
    std::vector<dicom_group_element> sq_data;
private:
    void assign(const dicom_group_element& rhs)
    {
        std::copy(rhs.gel,rhs.gel+8,gel);
        data = rhs.data;
        sq_data = rhs.sq_data;
    }
    bool flag_contains(const char* flag,unsigned int flag_size)
    {
        for (unsigned int index = 0; index < flag_size; ++index)
        {
            char lb = flag[index << 1];
            char hb = flag[(index << 1)+1];
            if (lt0 == lb && lt1 == hb)
                return true;
        }
        return false;
    }

public:
    dicom_group_element(void) {}
    dicom_group_element(const dicom_group_element& rhs)
    {
        assign(rhs);
    }
    const dicom_group_element& operator=(const dicom_group_element& rhs)
    {
        assign(rhs);
        return *this;
    }

    bool read(std::ifstream& in,transfer_syntax_type transfer_syntax,bool pause_at_image = true)
    {
        if (!in.read(gel,8))
            return false;
        // group 0002 is in Little Endian Implicit VR
        if(group == 0x0002)
            transfer_syntax = lee;
        if(transfer_syntax == bee)
        {
            change_endian(group);
            change_endian(element);
        }
        unsigned int read_length = length;
        bool is_explicit_vr = (transfer_syntax == bee || transfer_syntax == lee);
        bool is_sq = false;
        // SQ related Data Elements treated as implicit VR
        // http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_7.5.html
        if(group == 0xFFFE && (element == 0xE000 || element == 0xE00D || element == 0xE0DD))
        {
            is_explicit_vr = false;
            is_sq = true;
        }
        if(is_explicit_vr)
        {
            // http://dicom.nema.org/dicom/2013/output/chtml/part05/chapter_7.html#sect_7.1.2
            if (flag_contains(dicom_long_flag,6)) // Data Element with Explicit VR of OB, OW, OF, SQ, UT or UN
            {
                if (!in.read((char*)&read_length,4))
                    return false;
                if(transfer_syntax == bee)
                    change_endian(read_length);
            }
            else
            {
                if(transfer_syntax == bee)
                    change_endian(new_length);
                read_length = new_length;
            }
        }
        else
        {
            if(transfer_syntax == bee)
                change_endian(read_length);
        }
        // Handle image pixel data
        // http://dicom.nema.org/Dicom/2013/output/chtml/part05/sect_A.4.html
        //
        if(group == 0x7FE0 && element == 0x0010)
        {
            // encapsulated pixels
            if(read_length == 0xFFFFFFFF)
            {
                do{
                    // usually has one or more (FFFE,00E0)
                    in.read(gel,8);
                    // There could be dummy empty storage of (FFFE,00E0)
                    if(length <= 16)
                    {
                        char buf[16];
                        if(length)
                            in.read(buf,length);
                    }
                    else
                        return false;
                }while(in);
                length = 0;
                return false;
            }
            else
            {
                length = read_length;
                if(pause_at_image)
                    return false;
            }
        }
        if (read_length == 0xFFFFFFFF)
            read_length = 0;
        if (read_length)
        {
            // handle SQ here
            // http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_7.5.html
            if(is_sq ||
               (is_explicit_vr && lt0 == 'S' && lt1 == 'Q'))
            {
                size_t sq_end_pos = in.tellg();
                sq_end_pos += read_length;
                while(in && in.tellg() < sq_end_pos)
                {
                    dicom_group_element new_ge;
                    new_ge.read(in,transfer_syntax,false);
                    sq_data.push_back(new_ge);
                }
            }
            else
            {
                data.resize(read_length);
                in.read((char*)&*(data.begin()),read_length);
            }
            if(transfer_syntax == bee)
            {
                if (is_float()) // float
                    change_endian<float>(&*data.begin(),data.size()/sizeof(float));
                if (is_double()) // double
                    change_endian<double>(&*data.begin(),data.size()/sizeof(double));
                if (is_int16()) // uint16type
                    change_endian<short>(&*data.begin(),data.size()/sizeof(short));
                if (is_int32() && data.size() >= 4)
                    change_endian<int>(&*data.begin(),data.size()/sizeof(int));
            }
        }
        return !(!in);
    }

    unsigned int get_order(void) const
    {
        unsigned int order = group;
        order <<= 16;
        order |= element;
        return order;
    }
    const std::vector<unsigned char>& get(void) const
    {
        return data;
    }
    unsigned short get_vr(void) const
    {
        return vr;
    }

    bool is_string(void) const
    {
        return (lt0 == 'D' ||  // DA DS DT
                lt0 == 'P' ||  // PN
                lt0 == 'T' ||  // TM
                lt0 == 'L' ||  // LO LT
                lt1 == 'I' ||  // UI
                lt1 == 'H' ||  // SH
                (lt0 != 'A' && lt1 == 'T') || // ST UT LT
                (lt0 == 'A' && lt1 == 'E') || // AE
                ((lt0 == 'A' || lt0 == 'C' || lt0 == 'I') && lt1 == 'S'));//AS CS IS
    }
    bool is_int16(void) const
    {
        return (lt0 == 'A' && lt1 == 'T') ||
                (lt0 == 'O' && lt1 == 'W') ||
                (lt0 == 'S' && lt1 == 'S') ||
                (lt0 == 'U' && lt1 == 'S');
    }
    bool is_int32(void) const
    {
        return (lt0 == 'S' && lt1 == 'L') ||
                (lt0 == 'U' && lt1 == 'L');
    }
    bool is_float(void) const
    {
        //FL
        return (lt0 == 'F' && lt1 == 'L') || (lt0 == 'O' && lt1 == 'F');
    }
    bool is_double(void) const
    {
        //FD
        return (lt0 == 'F' && lt1 == 'D');
    }

    template<typename value_type>
    void get_value(std::vector<value_type>& value) const
    {
        if(data.empty())
            return;
        if (is_float() && data.size() >= 4) // float
        {
            const float* iter = (const float*)&*data.begin();
            for (unsigned int index = 3;index < data.size();index += 4,++iter)
                value.push_back(*iter);
            return;
        }
        if (is_double() && data.size() >= 8) // double
        {
            const double* iter = (const double*)&*data.begin();
            for (unsigned int index = 7;index < data.size();index += 8,++iter)
                value.push_back(*iter);
            return;
        }
        if (is_int16() && data.size() >= 2)
        {
            for (unsigned int index = 1;index < data.size();index+=2)
                value.push_back(*(const short*)&*(data.begin()+index-1));
            return;
        }
        if (is_int32() && data.size() == 4)
        {
            for (unsigned int index = 3;index < data.size();index+=4)
                value.push_back(*(const int*)&*(data.begin()+index-3));
            return;
        }
        if(is_string())
            std::copy(data.begin(),data.end(),std::back_inserter(value));
    }
    template<typename value_type>
    void get_value(value_type& value) const
    {
        if(data.empty())
            return;
        if constexpr (std::is_fundamental_v<value_type>)
        {
            if (is_float() && data.size() >= 4) // float
            {
                value = value_type(*(const float*)&*data.begin());
                return;
            }
            if (is_double() && data.size() >= 8) // double
            {
                value = value_type(*(const double*)&*data.begin());
                return;
            }
            if (is_int16() && data.size() >= 2) // uint16type
            {
                value = value_type(*(const short*)&*data.begin());
                return;
            }
            if (is_int32() && data.size() >= 4)
            {
                value = value_type(*(const int*)&*data.begin());
                return;
            }
        }
        bool is_ascii = true;
        if(!is_string())
        for (unsigned int index = 0;index < data.size() && (data[index] || index <= 2);++index)
            if (!::isprint(data[index]))
            {
                is_ascii = false;
                break;
            }
        if (is_ascii)
        {
            if constexpr(std::is_same_v<value_type, std::string>)
                value = std::string(data.begin(),data.end());
            else
            {
                std::string str(data.begin(),data.end());
                str.push_back(0);
                std::istringstream in(str);
                in >> value;
            }
            return;
        }
        if constexpr (std::is_fundamental_v<value_type>)
        {
            if (data.size() == 2) // uint16type
            {
                value = value_type(*(const short*)&*data.begin());
                return;
            }
            if (data.size() == 4)
            {
                value = value_type(*(const int*)&*data.begin());
                return;
            }
            if (data.size() == 8)
            {
                value = value_type(*(const double*)&*data.begin());
                return;
            }
        }
    }

    template<typename stream_type>
    void operator>> (stream_type& out) const
    {
        if(!sq_data.empty())
        {
            out << sq_data.size() << " SQ items";
            return;
        }
        if (data.empty())
        {
            out << "(null)";
            return;
        }
        if (is_float() && data.size() >= 4) // float
        {
            const float* iter = (const float*)&*data.begin();
            for (unsigned int index = 3;index < data.size();index += 4,++iter)
                out << *iter << " ";
            return;
        }
        if (is_double() && data.size() >= 8) // double
        {
            const double* iter = (const double*)&*data.begin();
            for (unsigned int index = 7;index < data.size();index += 8,++iter)
                out << *iter << " ";
            return;
        }
        if (is_int16() && data.size() >= 2)
        {
            for (unsigned int index = 1;index < data.size();index+=2)
                out << *(const short*)&*(data.begin()+index-1) << " ";
            return;
        }
        if (is_int32() && data.size() == 4)
        {
            for (unsigned int index = 3;index < data.size();index+=4)
                out << *(const int*)&*(data.begin()+index-3) << " ";
            return;
        }
        bool is_ascii = true;
        if (!is_string()) // String
        for (unsigned int index = 0;index < data.size() && (data[index] || index <= 2);++index)
            if (!::isprint(data[index]))
            {
            is_ascii = false;
            break;
            }
        if (is_ascii)
        {
            for (unsigned int index = 0;index < data.size();++index)
            {
                char ch = data[index];
                if (!ch)
                    break;
                out << ch;
            }
            return;
        }
        out << data.size() << " bytes";
        if(data.size() == 8)
            out << ", double=" << *(double*)&*data.begin() << " ";
        if(data.size() == 4)
            out << ", int=" << *(int*)&*data.begin() << ", float=" << *(float*)&*data.begin() << " ";
        if(data.size() == 2)
            out << ", short=" << *(short*)&*data.begin() << " ";
        return;
    }

};

struct dicom_csa_header
{
    char name[64];
    int vm;
    char vr[4];
    int syngodt;
    int nitems;
    int xx;
};

class dicom_csa_data
{
private:
    dicom_csa_header header;
    std::vector<std::string> vals;
    void assign(const dicom_csa_data& rhs)
    {
        std::copy(rhs.header.name,rhs.header.name+64,header.name);
        std::copy(rhs.header.vr,rhs.header.vr+4,header.vr);
        header.vm = rhs.header.vm;
        header.syngodt = rhs.header.syngodt;
        header.nitems = rhs.header.nitems;
        header.xx = rhs.header.xx;
        vals = rhs.vals;
    }
public:
    dicom_csa_data(void) {}
    dicom_csa_data(const dicom_csa_data& rhs)
    {
        assign(rhs);
    }
    const dicom_csa_data& operator=(const dicom_csa_data& rhs)
    {
        assign(rhs);
        return *this;
    }
    bool read(const std::vector<unsigned char>& data,unsigned int& from)
    {
        if (from + sizeof(dicom_csa_header) >= data.size())
            return false;
        std::copy(data.begin() + from,data.begin() + from + sizeof(dicom_csa_header),(char*)&header);
        from += sizeof(dicom_csa_header);
        int xx[4];
        for (int index = 0; index < header.nitems; ++index)
        {
            if (from + sizeof(xx) >= data.size())
                return false;
            std::copy(data.begin() + from,data.begin() + from + sizeof(xx),(char*)xx);
            from += sizeof(xx);
            if (from + xx[1] >= data.size())
                return false;
            if (xx[1])
                vals.push_back(std::string(data.begin() + from,data.begin() + from + xx[1]-1));
            from += xx[1] + (4-(xx[1]%4))%4;
        }
        return true;
    }
    void write_report(std::string& lines) const
    {
        std::ostringstream out;
        out << header.name << ":" << header.vm << ":" << header.vr << ":" << header.syngodt << ":" << header.nitems << "=";
        for (unsigned int index = 0; index < vals.size(); ++index)
            out << vals[index] << " ";
        lines += out.str();
        lines += "\n";
    }
    const char* get_value(unsigned int index) const
    {
        if (index < vals.size())
            return &*vals[index].begin();
        return 0;
    }
    const char* get_name(void) const
    {
        return header.name;
    }
};

class dicom
{
private:
    std::shared_ptr<std::ifstream> input_io;
    unsigned int image_size = 0;
    transfer_syntax_type transfer_syntax;
public:
    std::string encoding;
    mutable bool is_compressed = false;
    mutable std::vector<char> compressed_buf;
    unsigned int buf_size = 0;
public:
    std::vector<dicom_group_element> data;
    std::map<unsigned int,unsigned int> ge_map;
    std::map<std::string,unsigned int> csa_map;
    std::vector<dicom_csa_data> csa_data;
    bool is_mosaic = false;
    bool is_multi_frame = false;
    bool is_big_endian = false;
private:
    void assign(const dicom& rhs)
    {
        ge_map = rhs.ge_map;
        csa_map = rhs.csa_map;
        for (unsigned int index = 0;index < rhs.data.size();index++)
            data.push_back(rhs.data[index]);
        for (unsigned int index = 0;index < rhs.csa_data.size();index++)
            csa_data.push_back(rhs.csa_data[index]);
    }
    template<typename iterator_type>
    void handle_mosaic(iterator_type image_buffer,
                       unsigned int mosaic_width,
                       unsigned int mosaic_height,
                       unsigned int w,unsigned int h) const
    {
        typedef typename std::iterator_traits<iterator_type>::value_type pixel_type;
        // number of image in mosaic
        unsigned int mosaic_size = mosaic_width*mosaic_height;
        std::vector<pixel_type> data(w*h);
        std::copy(image_buffer,image_buffer+data.size(),data.begin());
        // rearrange mosaic

        unsigned int mosaic_col_count = w/mosaic_width;
        unsigned int mosaic_line_size = mosaic_size*mosaic_col_count;


        const pixel_type* slice_end = &*data.begin() + data.size();
        for (const pixel_type* slice_band_pos = &*data.begin(); slice_band_pos < slice_end; slice_band_pos += mosaic_line_size)
        {
            const pixel_type* slice_pos_end = slice_band_pos + w;
            for (const pixel_type* slice_pos = slice_band_pos; slice_pos < slice_pos_end; slice_pos += mosaic_width)
            {
                const pixel_type* slice_line_end = slice_pos + mosaic_line_size;
                for (const pixel_type* slice_line = slice_pos; slice_line < slice_line_end; slice_line += w,image_buffer += mosaic_width)
                    std::copy(slice_line,slice_line+mosaic_width,image_buffer);
            }
        }
    }
    static void clean_name(std::string& s)
    {
        std::string key_chars = "\\/:?\"<>|^";
        for (size_t i = 0;i < s.length();++i)
            if(key_chars.find(s[i]) != std::string::npos)
                s[i] = '_';
    }
public:
    dicom(void):transfer_syntax(lee) {}
    dicom(const dicom& rhs)
    {
        assign(rhs);
    }
    const dicom& operator=(const dicom& rhs)
    {
        assign(rhs);
        return *this;
    }
public:
    bool load_from_file(const std::string& file_name)
    {
        return load_from_file(file_name.c_str());
    }
    template<typename char_type>
    bool load_from_file(const char_type* file_name)
    {
        ge_map.clear();
        data.clear();
        transfer_syntax = lee;
        input_io.reset(new std::ifstream(file_name,std::ios::binary));
        if (!(*input_io))
            return false;
        input_io->seekg(128);
        unsigned int dicom_mark = 0;
        input_io->read((char*)&dicom_mark,4);
        if (dicom_mark != 0x4d434944) //DICM
        {
            // switch to another DICOM format
            input_io->seekg(0,std::ios::beg);
            input_io->read((char*)&dicom_mark,4);
            if(dicom_mark != 0x00050008 &&
               dicom_mark != 0x00000008)
                return false;
            // some DICOM start with lei as default
            {
                unsigned char VR[2];
                input_io->read((char*)&VR,2);
                if(VR[1] == 0)
                    transfer_syntax = lei;
            }
            input_io->seekg(0,std::ios::beg);
        }
        while (*input_io)
        {
            dicom_group_element ge;
            if (!ge.read(*input_io,transfer_syntax))
            {
                if (!(*input_io))
                    return true;
                {
                    std::string image_type;
                    is_multi_frame = get_int(0x0028,0x0008) > 1 || (image_size > width()*height());   // multiple frame (new version)
                    is_mosaic = get_int(0x0019,0x100A) > 1 || (get_text(0x0008,0x0008,image_type) && image_type.find("MOSAIC") != std::string::npos);
                }
                if(is_compressed)
                {
                    buf_size = ge.length;
                    is_big_endian = false;
                    return true;
                }
                image_size = ge.length;
                is_big_endian = (transfer_syntax == bee);
                return true;
            }

            // detect transfer syntax at 0x0002,0x0010
            if (ge.group == 0x0002 && ge.element == 0x0010)
            {
                if(std::string((char*)&*ge.data.begin()) == std::string("1.2.840.10008.1.2"))
                    transfer_syntax = lei;//Little Endian Implicit
                else
                if(std::string((char*)&*ge.data.begin()) == std::string("1.2.840.10008.1.2.1"))
                    transfer_syntax = lee;//Little Endian Explicit
                else
                if(std::string((char*)&*ge.data.begin()) == std::string("1.2.840.10008.1.2.2"))
                    transfer_syntax = bee;//Big Endian Explicit
                else
                {
                    is_compressed = true;
                    encoding.resize(ge.data.size());
                    std::copy(ge.data.begin(),ge.data.end(),encoding.begin());
                    /*
                    1.2.840.10008.1.2.1.99 (Deflated Explicit VR Little Endian)
                    1.2.840.10008.1.2.4.50 (JPEG Baseline (Process 1) Lossy JPEG 8-bit)
                    1.2.840.10008.1.2.4.51 (JPEG Baseline (Processes 2 & 4) Lossy JPEG 12-bit)
                    1.2.840.10008.1.2.4.57 (JPEG Lossless, Nonhierarchical (Processes 14))
                    1.2.840.10008.1.2.4.70 (JPEG Lossless, Nonhierarchical (Processes 14 [Selection 1]))
                    1.2.840.10008.1.2.4.80 (JPEG-LS Image Compression (Lossless Only))
                    1.2.840.10008.1.2.4.81 (JPEG-LS Image Compression)
                    1.2.840.10008.1.2.4.90 (JPEG 2000 Image Compression (Lossless Only))
                    1.2.840.10008.1.2.4.91 (JPEG 2000 Image Compression)
                    1.2.840.10008.1.2.5 (RLE Lossless)
                    */
                }
            }
            // Deal with CSA
            if (ge.group == 0x0029 && (ge.element == 0x1010 || ge.element == 0x1020) && ge.get().size() >= 4)
            {
                std::string SV10(ge.get().begin(),ge.get().begin()+4);
                if (SV10 == "SV10")
                {
                    int count = *(int*)&ge.get()[8];
                    if (count <= 128 && count >= 0)
                    {
                        unsigned int pos = 16;
                        for (unsigned int index = 0; index < (unsigned int)count && pos < ge.get().size(); ++index)
                        {
                            dicom_csa_data csa;
                            if (!csa.read(ge.get(),pos))
                                break;
                            csa_data.push_back(csa);
                            csa_map[csa_data.back().get_name()] = (unsigned int)(csa_data.size()-1);
                        }
                    }
                }

            }
            if(ge.group == 0xFFFE && ge.data.empty())
                continue;
            auto& item = ge_map[ge.get_order()];
            if(item == 0) // check if there is duplicate group element
                item = (unsigned int)(data.size());
            data.push_back(ge);
        }
        return true;
    }

    const char* get_csa_data(const std::string& name,unsigned int index) const
    {
        std::map<std::string,unsigned int>::const_iterator iter = csa_map.find(name);
        if (iter == csa_map.end())
            return 0;
        return csa_data[iter->second].get_value(index);
    }

    const unsigned char* get_data(unsigned short group,unsigned short element,unsigned int& length) const
    {
        std::map<unsigned int,unsigned int>::const_iterator iter =
                ge_map.find(((unsigned int)group << 16) | (unsigned int)element);
        if (iter == ge_map.end())
        {
            length = 0;
            return 0;
        }
        length = (unsigned int)data[iter->second].get().size();
        if (!length)
            return 0;
        return (const unsigned char*)&*data[iter->second].get().begin();
    }

    bool get_text(unsigned short group,unsigned short element,std::string& result) const
    {
        unsigned int length = 0;
        const char* text = (const char*)get_data(group,element,length);
        if (!text)
            return false;
        result = std::string(text,text+length);
        return true;
    }
    std::string get_text(unsigned short group,unsigned short element) const
    {
        unsigned int length = 0;
        const char* text = (const char*)get_data(group,element,length);
        if (!text)
            return std::string();
        return std::string(text,text+length);
    }

    bool get_text_all(unsigned short group,unsigned short element,std::string& result) const
    {
        std::string re;
        for(int i = 0;i < data.size();++i)
            if(data[i].group == group && data[i].element == element)
            {
                auto& buf = data[i].get();
                re += std::string(buf.begin(),buf.end());
            }
        result = re;
        return !re.empty();
    }

    template<typename value_type>
    bool get_value(unsigned short group,unsigned short element,value_type& value) const
    {
        std::map<unsigned int,unsigned int>::const_iterator iter =
                ge_map.find(((unsigned int)group << 16) | (unsigned int)element);
        if (iter == ge_map.end())
            return false;
        data[iter->second].get_value(value);
        return true;
    }
    template<typename value_type>
    bool get_frame_values(unsigned short group,unsigned short element,std::vector<value_type>& values) const
    {
        bool result = false;
        values.clear();
        unsigned int ge = ((unsigned int)group << 16) | (unsigned int)element;
        for(int i = 0;i < data.size();++i)
            if(data[i].get_order() == ge)
            {
                value_type t;
                data[i].get_value(t);
                values.push_back(t);
                result = true;
            }
        return result;
    }
    unsigned int get_int(unsigned short group,unsigned short element) const
    {
        unsigned int value = 0;
        get_value(group,element,value);
        return value;
    }
    float get_float(unsigned short group,unsigned short element) const
    {
        float value = 0.0;
        get_value(group,element,value);
        return value;
    }
    double get_double(unsigned short group,unsigned short element) const
    {
        double value = 0.0;
        get_value(group,element,value);
        return value;
    }
    void get_voxel_size(tipl::vector<3,float>& voxel_size) const
    {
        std::string slice_dis;
        if (get_text(0x0018,0x0088,slice_dis) || get_text(0x0018,0x0050,slice_dis))
            std::istringstream(slice_dis) >> voxel_size[2];
        else
            voxel_size[2] = 1.0;

        std::string pixel_spacing;
        if (get_text(0x0028,0x0030,pixel_spacing))
        {
            std::replace(pixel_spacing.begin(),pixel_spacing.end(),'\\',' ');
            std::istringstream(pixel_spacing) >> voxel_size[1] >> voxel_size[0];
        }
        else
            voxel_size[0] = voxel_size[1] = voxel_size[2];
    }

    /**
    The DICOM attribute (0020,0037) "Image Orientation (Patient)" gives the
    orientation of the x- and y-axes of the image data in terms of 2 3-vectors.
    The first vector is a unit vector along the x-axis, and the second is
    along the y-axis.
    */
    template<typename vector_type>
    bool get_image_row_orientation(vector_type image_row_orientation) const
    {
        //float image_row_orientation[3];
        std::string image_orientation;
        if (!get_text(0x0020,0x0037,image_orientation) &&
                !get_text(0x0020,0x0035,image_orientation))
            return false;
        std::replace(image_orientation.begin(),image_orientation.end(),'\\',' ');
        std::istringstream(image_orientation)
        >> image_row_orientation[0]
        >> image_row_orientation[1]
        >> image_row_orientation[2];
        return true;
    }
    template<typename vector_type>
    bool get_image_col_orientation(vector_type image_col_orientation) const
    {
        //float image_col_orientation[3];
        float temp;
        std::string image_orientation;
        if (!get_text(0x0020,0x0037,image_orientation) &&
                !get_text(0x0020,0x0035,image_orientation))
            return false;
        std::replace(image_orientation.begin(),image_orientation.end(),'\\',' ');
        std::istringstream(image_orientation)
        >> temp >> temp >> temp
        >> image_col_orientation[0]
        >> image_col_orientation[1]
        >> image_col_orientation[2];
        return true;
    }
    template<typename vector_type>
    bool get_left_upper_pos(vector_type lp_pos) const
    {
        std::string pos;
        if (!get_text(0x0020,0x0032,pos))
            return false;
        std::replace(pos.begin(),pos.end(),'\\',' ');
        std::istringstream(pos) >> lp_pos[0] >> lp_pos[1] >> lp_pos[2];
        return true;
    }

    template<typename vector_type>
    bool get_image_orientation(vector_type orientation_matrix) const
    {
        if(!get_image_row_orientation(orientation_matrix) ||
           !get_image_col_orientation(orientation_matrix+3))
            return false;
        // get the slice direction
        orientation_matrix[6] =
            (orientation_matrix[1] * orientation_matrix[5])-
            (orientation_matrix[2] * orientation_matrix[4]);
        orientation_matrix[7] =
            (orientation_matrix[2] * orientation_matrix[3])-
            (orientation_matrix[0] * orientation_matrix[5]);
        orientation_matrix[8] =
            (orientation_matrix[0] * orientation_matrix[4])-
            (orientation_matrix[1] * orientation_matrix[3]);

        // the slice ordering is always increamental
        if (orientation_matrix[6] + orientation_matrix[7] + orientation_matrix[8] < 0) // no flip needed
        {
            orientation_matrix[6] = -orientation_matrix[6];
            orientation_matrix[7] = -orientation_matrix[7];
            orientation_matrix[8] = -orientation_matrix[8];
        }
        return true;
    }
    float get_slice_location(void) const
    {
        std::string slice_location;
        if (!get_text(0x0020,0x1041,slice_location))
            return 0.0;
        float data;
        std::istringstream(slice_location) >> data;
        return data;
    }
    float get_te(void) const
    {
        return get_float(0x0018,0x0081);
    }
    bool get_btable(float& bvalue,float& bx,float& by,float& bz)
    {
        if(!csa_map.empty())
        {
            const char* b_value_ptr = get_csa_data("B_value",0);
            const char* bx_ptr = get_csa_data("DiffusionGradientDirection",0);
            const char* by_ptr = get_csa_data("DiffusionGradientDirection",1);
            const char* bz_ptr = get_csa_data("DiffusionGradientDirection",2);
            if (b_value_ptr && bx_ptr && by_ptr && bz_ptr)
            {
                std::istringstream(std::string(b_value_ptr)) >> bvalue;
                std::istringstream(std::string(bx_ptr)) >> bx;
                std::istringstream(std::string(by_ptr)) >> by;
                std::istringstream(std::string(bz_ptr)) >> bz;
                return true;
            }
        }

        if(!get_value(0x0018,0x9087,bvalue) &&
           !get_value(0x0019,0x100C,bvalue) &&
           !get_value(0x0043,0x1039,bvalue) &&
           !get_value(0x0065,0x1009,bvalue))
            return false;

        if(bvalue == 0.0f)
        {
            bx = by = bz = 0.0f;
            return true;
        }

        std::vector<double> bvec;
        if(get_value(0x0018,0x9089,bvec) ||
           get_value(0x0019,0x100E,bvec) ||
           get_value(0x0019,0x1027,bvec) ||
           get_value(0x0065,0x1037,bvec))
        {
            if(bvec.size() < 3)
                return false;
            bx = bvec[0];by = bvec[1];bz = bvec[2];
            if(bvec.size() >= 6 && bvec[0] == 0.0)
            {
                bx = by = bz = 0.0;
                if (bvec[3] != 0.0)
                {
                    by = bvec[3];
                    bz = bvec[4];
                }
                else
                    bz = bvec[5];
            }
            return true;
        }

        //GE
        if(get_value(0x0019,0x10BB,bx) &&
           get_value(0x0019,0x10BC,by) &&
           get_value(0x0019,0x10BD,bz))
        {
            bz = -bz;
            bvalue *= std::sqrt(bx*bx+by*by+bz*bz);
            return true;
        }

        std::string b_str;
        // TOSHIBU uses b-value string e.g.  b=2000(0.140,0.134,-0.981)
        if(get_text(0x0020,0x4000,b_str))
        {
            std::replace(b_str.begin(),b_str.end(),'(',' ');
            std::replace(b_str.begin(),b_str.end(),')',' ');
            std::replace(b_str.begin(),b_str.end(),',',' ');
            std::istringstream in(b_str);
            in >> bvalue >> bx >> by >> bz;
            return true;
        }
        return false;
    }

    void get_patient(std::string& info)
    {
        std::string date,gender,age,id;
        date = gender = age = id = "_";
        get_text(0x0008,0x0022,date);
        get_text(0x0010,0x0040,gender);
        get_text(0x0010,0x1010,age);
        get_text(0x0010,0x0010,id);
        using namespace std;
        gender.erase(remove(gender.begin(),gender.end(),' '),gender.end());
        id.erase(remove(id.begin(),id.end(),' '),id.end());
        clean_name(id);
        info = date;
        info += "_";
        info += gender;
        info += age;
        info += "_";
        info += id;
    }
    void get_sequence_id(std::string& seq)
    {
        get_text(0x0008,0x103E,seq);
        using namespace std;
        seq.erase(remove(seq.begin(),seq.end(),' '),seq.end());
        clean_name(seq);
    }
    void get_sequence_num(std::string& info)
    {
        std::string series_num;
        get_text(0x0020,0x0011,series_num);
        using namespace std;
        series_num.erase(remove(series_num.begin(),series_num.end(),' '),series_num.end());
        if (series_num.size() == 1)
        {
            info = std::string("0");
            info += series_num;
        }
        else
            info = series_num;
    }
    void get_sequence(std::string& info)
    {
        std::string series_num,series_des;
        get_sequence_num(series_num);
        get_sequence_id(series_des);
        info = series_num;
        info += "_";
        info += series_des;
    }
    std::string get_image_num(void)
    {
        std::string image_num;
        get_text_all(0x0020,0x0013,image_num);
        using namespace std;
        if(!image_num.empty())
            image_num.erase(remove(image_num.begin(),image_num.end(),' '),image_num.end());
        return image_num;
    }

    void get_image_name(std::string& info)
    {
        std::string series_des;
        series_des = "_";
        get_sequence_id(series_des);
        info = series_des;
        info += "_i";
        info += get_image_num();
        info += ".dcm";
    }

    unsigned int width(void) const
    {
        return get_int(0x0028,0x0011);
    }

    unsigned int height(void) const
    {
        return get_int(0x0028,0x0010);
    }

    unsigned int frame_num(void) const
    {
        return get_int(0x0028,0x0008);
    }

    unsigned int get_bit_count(void) const
    {
        return get_int(0x0028,0x0100);
    }
    unsigned int is_signed(void) const
    {
        return get_int(0x0028,0x0103);
    }

    void get_image_dimension(tipl::shape<2>& geo) const
    {
        geo[0] = width();
        geo[1] = height();
    }

    void get_image_dimension(tipl::shape<3>& geo) const
    {
        geo[0] = width();
        geo[1] = height();
        geo[2] = 1;

        const char* mosaic = get_csa_data("NumberOfImagesInMosaic",0);
        if(mosaic)
            geo[2] = std::stoi(mosaic);
        else
            geo[2] = get_int(0x0019,0x100A);
        if(geo[2])
        {
            geo[0] = int(width()/std::ceil(std::sqrt(geo[2])));
            geo[1] = int(height()/std::ceil(std::sqrt(geo[2])));
        }
        else
        {
            if(get_bit_count() && geo[0] && geo[1])
                geo[2] = image_size/geo[0]/geo[1]/(get_bit_count()/8);
            else
                geo[2] = 1;
        }
    }


    template<typename pointer_type>
    void save_to_buffer(pointer_type ptr,unsigned int pixel_count) const
    {
        typedef typename std::iterator_traits<pointer_type>::value_type value_type;
        if(is_compressed)
        {
            compressed_buf.resize(buf_size);
            input_io->read((char*)&*(compressed_buf.begin()),buf_size);
            if(encoding == "1.2.840.10008.1.2.4.70")
            {
                std::vector<unsigned char> buf;
                int X,Y,bits,frames;
                if(decode_1_2_840_10008_1_2_4_70((unsigned char*)&*(compressed_buf.begin()),buf_size,buf,&X,&Y,&bits,&frames))
                {
                    if(bits == 8)
                        std::copy(buf.begin(),buf.begin()+std::min<size_t>(pixel_count,buf.size()),ptr);
                    if(bits == 16)
                        std::copy(reinterpret_cast<const short*>(buf.data()),
                                  reinterpret_cast<const short*>(buf.data()) + +std::min<size_t>(pixel_count,buf.size() >> 1),ptr);
                    compressed_buf.clear();
                    is_compressed = false;
                }
            }
            return;
        }

        if(sizeof(value_type) == get_bit_count()/8)
            input_io->read((char*)&*ptr,pixel_count*sizeof(value_type));
        else
        {
            std::vector<char> data(pixel_count*get_bit_count()/8);
            input_io->read((char*)&(data[0]),data.size());
            switch (get_bit_count()) //bit count
            {
            case 8://DT_UNSIGNED_CHAR 2
                std::copy((const unsigned char*)&(data[0]),(const unsigned char*)&(data[0])+pixel_count,ptr);
                return;
            case 16://DT_SIGNED_SHORT 4
                if(is_big_endian)
                    change_endian((unsigned short*)&(data[0]),pixel_count);
                if(is_signed())
                    std::copy((const short*)&(data[0]),(const short*)&(data[0])+pixel_count,ptr);
                else
                    std::copy((const unsigned short*)&(data[0]),(const unsigned short*)&(data[0])+pixel_count,ptr);
                return;
            case 32://DT_SIGNED_INT 8
                if(is_big_endian)
                    change_endian((unsigned int*)&(data[0]),pixel_count);
                if(is_signed())
                    std::copy((const int*)&(data[0]),(const int*)&(data[0])+pixel_count,ptr);
                else
                    std::copy((const unsigned int*)&(data[0]),(const unsigned int*)&(data[0])+pixel_count,ptr);
                return;
            case 64://DT_DOUBLE 64
                if(is_big_endian)
                    change_endian((double*)&(data[0]),pixel_count);
                std::copy((const double*)&(data[0]),(const double*)&(data[0])+pixel_count,ptr);
                return;
            }
        }
    }

    template<typename image_type>
    void save_to_image(image_type& out) const
    {
        if(!input_io.get() || !(*input_io))
            return;
        tipl::shape<image_type::dimension> geo;
        get_image_dimension(geo);
        if(is_mosaic)
        {
            unsigned short slice_num = geo[2];
            geo[2] = width()*height()/geo[0]/geo[1];
            out.resize(geo);
            save_to_buffer(out.begin(),(unsigned int)out.size());

            if(geo[2] == 1)// find mosaic pattern by numerical approach
            {
                unsigned int mosaic_factor = 0;

                // approach 1: row sum = 0 is the separator
                for(int y = 10,y_pos = 10*geo[0];y < geo[1];++y,y_pos += geo[0])
                {
                    int m = std::round((float)geo[1]/(float)y);
                    if(m > 20 || m < 4 || width() % m != 0)
                        continue;
                    int sum_x = std::accumulate(out.begin()+y_pos,out.begin()+y_pos+geo[0],(int)0);
                    int sum_y = 0;
                    for(int i = y;i < out.size();i += geo[0])
                        sum_y += out[i];
                    if(sum_x == 0 || sum_y == 0)
                    {
                        mosaic_factor = m;
                        break;
                    }
                }
                // approach 2: column sum smoothed peaks.
                if(!mosaic_factor)
                {
                    std::vector<float> profile_x(geo[0]),new_profile_x(geo[0]);
                    for(int x = 0;x < profile_x.size();++x)
                    {
                        for(int y = 0,y_pos = 0;y < geo[1];++y,y_pos +=geo[0])
                            profile_x[x] += out[x+y_pos];
                    }
                    for(int iter = 0;iter < 128;++iter)
                    {
                        new_profile_x[0] = profile_x[0];
                        new_profile_x.back() = profile_x.back();
                        for(int x = 1;x+1 < profile_x.size();++x)
                            new_profile_x[x] = (profile_x[x-1] + profile_x[x] + profile_x[x+1])*0.333f;
                        new_profile_x.swap(profile_x);
                    }
                    for(int x = 1;x+1 < profile_x.size();++x)
                        if(profile_x[x-1] < profile_x[x] &&
                           profile_x[x+1] < profile_x[x])
                            ++mosaic_factor;
                    if(geo[0] % (mosaic_factor + 1) == 0)
                        mosaic_factor = mosaic_factor + 1;
                    if(geo[0] % (mosaic_factor - 1) == 0)
                        mosaic_factor = mosaic_factor - 1;
                }
                geo[0] /= mosaic_factor;
                geo[1] /= mosaic_factor;
                slice_num = mosaic_factor*mosaic_factor;
            }
            handle_mosaic(out.begin(),geo[0],geo[1],width(),height());
            geo[2] = slice_num;
            out.resize(geo);
        }
        else
        {
            out.resize(geo);
            save_to_buffer(out.begin(),(unsigned int)out.size());
        }
    }

    template<typename image_type>
    image_type& operator>>(image_type& source) const
    {
        save_to_image(source);
        return source;
    }
    template<typename image_type>
    image_type& operator<<(const image_type& source)
    {
        load_from_image(source);
        return source;
    }
    template<typename value_type>
    static bool get_value(const dicom_group_element& data,unsigned short group, unsigned short element,value_type& value)
    {
        if(data.group == group && data.element == element)
        {
            data.get_value(value);
            return true;
        }
        for(int i = 0;i < data.sq_data.size();++i)
            if(get_value(data.sq_data[i],group,element,value))
                return true;
        return false;
    }
    static void get_report(const std::vector<dicom_group_element>& data,std::string& report,bool item_tag = false)
    {
        std::ostringstream out;
        for (int i = 0;i < data.size();++i)
        {
            std::string group_element_str;
            {
                std::ostringstream ge_out;
                if(data[i].group == 0xFFFE)
                    ge_out << "item";
                else
                    ge_out << "("  << std::setw( 4 ) << std::setfill( '0' ) << std::hex << std::uppercase << data[i].group
                           << ","  << std::setw( 4 ) << std::setfill( '0' ) << std::hex << std::uppercase << data[i].element << ")";                                                        ;
                if(item_tag)
                    ge_out << "[" << i << "].";
                group_element_str = ge_out.str();
            }

            if(!data[i].sq_data.empty())
            {
                std::string nest_report;
                get_report(data[i].sq_data,nest_report,true);
                std::istringstream nest_in(nest_report);
                std::string line;
                while(std::getline(nest_in,line))
                    out << group_element_str << line << std::endl;
                continue;
            }

            out << group_element_str << "=";

            if(data[i].data.empty())
            {
                out << "empty";
            }
            else
            {
                out << data[i].data.size() << " bytes ";
                if(data[i].data.empty())
                {
                    out << std::setw( 8 ) << std::setfill( '0' ) << std::hex << std::uppercase <<
                    data[i].length << " ";
                    out << std::dec;
                }
                else
                {
                    unsigned short vr = data[i].vr;
                    if((vr & 0xFF) && (vr >> 8))
                        out << (char)(vr & 0xFF) << (char)(vr >> 8) << " ";
                    else
                        out << "   ";
                    data[i] >> out;
                }
            }
            out << std::endl;
        }
        report += out.str();
    }

    const dicom& operator>>(std::string& report) const
    {
        get_report(data,report);
        for(unsigned int index = 0;index < csa_data.size();++index)
            csa_data[index].write_report(report);
        return *this;
    }
};

class dicom_volume{
public:
    std::vector<std::shared_ptr<dicom> > dicom_reader;
    float orientation_matrix[9];
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    uint8_t dim_order[3]; // used to rotate the volume to axial view
    uint8_t flip[3];        // used to rotate the volume to axial view
    std::string error_msg;

    void free_all(void)
    {
        dicom_reader.clear();
    }
    void change_orientation(bool x,bool y,bool z)
    {
        bool xyz[3];
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
        for(int index = 0;index < 3;++index)
            if(xyz[dim_order[index]])
                flip[index] = !flip[index];
    }
public:
    ~dicom_volume(void){free_all();}
    const std::shared_ptr<dicom> get_dicom(unsigned int index) const{return dicom_reader[index];}
    const tipl::shape<3>& shape(void) const{return dim;}
    void get_voxel_size(tipl::vector<3,float>& voxel_size) const
    {
        voxel_size = vs;
    }
    template<typename vector_type>
    void get_image_row_orientation(vector_type image_row_orientation) const
    {
        std::copy(orientation_matrix,orientation_matrix+9,image_row_orientation);
    }
    bool load_from_files(const std::vector<std::string>& files)
    {
        if(files.empty())
            return false;
        free_all();
        std::vector<int> image_num;
        unsigned int w(0),h(0);
        for (unsigned int index = 0;index < files.size();++index)
        {
            std::shared_ptr<dicom> d(new dicom);
            if (!d->load_from_file(files[index]))
            {
                error_msg = "failed to read ";
                error_msg += files[index];
                return false;
            }
            if(index)
            {
                if(d->width() != w || d->height() != h)
                {
                    error_msg = "inconsistent image dimension at ";
                    error_msg += files[index];
                    return false;
                }
            }
            else
            {
                w = d->width();
                h = d->height();
            }

            // get image sequence
            image_num.push_back(0);
            std::istringstream(d->get_image_num()) >> image_num.back();
            dicom_reader.push_back(d);
        }

        if(files.size() == 1)
        {
            dicom_reader.front()->get_image_dimension(dim);
            dicom_reader.front()->get_voxel_size(vs);
            dicom_reader.front()->get_image_orientation(orientation_matrix);
        }
        else
        {        // sort dicom according to the image num
            {
                auto order = tipl::arg_sort(image_num.size(),[&](uint32_t i,uint32_t j){return image_num[i] < image_num[j];});
                std::vector<std::shared_ptr<dicom> > new_dicom_reader(order.size());
                for(size_t i = 0;i < order.size();++i)
                    new_dicom_reader[i] = dicom_reader[order[i]];
                new_dicom_reader.swap(dicom_reader);
            }
            dim = tipl::shape<3>(dicom_reader.front()->width(),
                                 dicom_reader.front()->height(),
                                 uint32_t(dicom_reader.size()));
            dicom_reader.front()->get_voxel_size(vs);
            dicom_reader.front()->get_image_orientation(orientation_matrix);
            if(vs[2] == 0.0f)
                vs[2] = std::fabs(dicom_reader[1]->get_slice_location()-
                                                  dicom_reader[0]->get_slice_location());
            // the last row of the orientation matrix should be derived from slice location
            // otherwise, could be flipped in the saggital slices
            {
                tipl::vector<3> pos1,pos2;
                dicom_reader[0]->get_left_upper_pos(pos1.begin());
                dicom_reader[1]->get_left_upper_pos(pos2.begin());
                if(pos1 == pos2)
                {
                    error_msg = "duplicated slices found.";
                    return false;
                }
                orientation_matrix[6] = pos2[0]-pos1[0];
                orientation_matrix[7] = pos2[1]-pos1[1];
                orientation_matrix[8] = pos2[2]-pos1[2];
            }
        }
        tipl::get_orientation(3,orientation_matrix,dim_order,flip);
        tipl::reorient_vector(vs,dim_order);
        tipl::reorient_matrix(orientation_matrix,dim_order,flip);
        return true;
    }
    template<typename image_type>
    void get_untouched_image(image_type& source) const
    {
        if(dicom_reader.empty())
            return;
        if(dicom_reader.size() == 1)
            *dicom_reader.front() >> source;
        else
        {
            source.resize(dim);
            for(size_t index = 0;index < dicom_reader.size();++index)
                dicom_reader[index]->save_to_buffer(&*source.begin()+index*dim.plane_size(),dim.plane_size());
        }
    }

    template<typename image_type>
    void save_to_image(image_type& I) const
    {
        typename image_type::buffer_type buffer;
        get_untouched_image(buffer);
        tipl::reorder(buffer,I,dim_order,flip); // to LPS
    }

    template<typename image_type>
    image_type& operator>>(image_type& I) const
    {
        save_to_image(I);
        return I;
    }
};

}

}
#endif
