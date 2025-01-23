# -*- coding: utf-8 -*-
"""
This is a Python translation of ParkSystemsTiffReader.m Matlab script

"""

import struct
import numpy as np


def ParkSystemsTiffReader(filename):


    with open(filename,'rb') as file:
        fbytes = file.read(3)
        fbytes.decode("utf-8") == 'II*'
        file.read(1)
        offset = struct.unpack('i',file.read(4))[0]
        file.seek(offset)
        nidf = struct.unpack('h',file.read(2))[0]

        idfs = {}    
    
        
        for k in range(0,nidf):
            idf = {'tag': struct.unpack('H',file.read(2))[0],
                   'type': struct.unpack('H',file.read(2))[0],
                    'count': struct.unpack('I',file.read(4))[0],
                  }
            
            if idf['type'] == 3:
                if idf['count'] > 2:
                    idf['value'] = struct.unpack('I',file.read(4))[0]
                else:
                    idf['value'] = struct.unpack('H',file.read(2))[0]
                    struct.unpack('H',file.read(2))[0]
            else:
                idf['value'] = struct.unpack('I',file.read(4))[0]
            
                 
                
            if idf['tag'] == 256 : 
                 idfs['ImageWidth'] = idf
            elif idf['tag'] == 257 :  
                 idfs['ImageLength'] = idf
            elif idf['tag'] == 258 :
                 idfs['BitsPerSample'] = idf
            elif idf['tag'] == 259 :
                 idfs['Compression'] = idf
            elif idf['tag'] == 262 :
                 idfs['PhotometricInterpretation'] = idf
            elif idf['tag'] == 273 :   
                 idfs['StripOffsets'] = idf
            elif idf['tag'] == 274 :
                 idfs['Orientation'] = idf
            elif idf['tag'] == 277 :
                 idfs['SamplesPerPixel'] = idf
            elif idf['tag'] == 278 :
                 idfs['RowsPerStrip'] = idf         
            elif idf['tag'] == 279 :
                 idfs['StripByteCounts'] = idf  
            elif idf['tag'] == 284 :
                 idfs['PlanerConfiguration'] = idf  
            elif idf['tag'] == 305 :
                 idfs['Software'] = idf 
            elif idf['tag'] == 306 :
                 idfs['DateTime'] = idf   
            elif idf['tag'] == 320 :
                 idfs['ColorMap'] = idf  
            elif idf['tag'] == 50432 :
                 idfs['PSIAMagicNumber'] = idf  
            elif idf['tag'] == 50433 :
                 idfs['PSIAVersion'] = idf  
            elif idf['tag'] == 50434 :
                 idfs['PSIAData'] = idf  
            elif idf['tag'] == 50435 :
                 idfs['PSIAHeader'] = idf  
            elif idf['tag'] == 50436 :
                 idfs['PSIACommnets'] = idf  
            elif idf['tag'] == 50437 :
                 idfs['PSIAProfileHeader'] = idf
            elif idf['tag'] == 50438 :
                 idfs['PSIASpectorscopyHeader'] = idf  
            elif idf['tag'] == 50439 :
                 idfs['PSIASpectorscopyData'] = idf  
            elif idf['tag'] == 50440 :
                 idfs['PSIAPddRegion'] = idf  
            else:
                 idfs['dummy'] = idf  
                
        
        if 'PSIAMagicNumber' in idfs: 
           if idfs['PSIAMagicNumber']['value'] == 235082497:
               # get colormap [320]
               file.seek(idfs['ColorMap']['value'],0) # 0='bof'
               #nidf = struct.unpack('H',file.read(2))[0]
               temp = np.asarray(struct.unpack('h'*768,file.read(768*2)))
               ColorMap = np.reshape(temp.astype(float),(256,3))
               # get Header [50435] 
               file.seek(idfs['PSIAHeader']['value'],0)
               
               Header = {'ImageType' : struct.unpack('i',file.read(4))[0]}
               
               buffer = np.asarray(struct.unpack('h'*32,file.read(32*2)))
               inds = [i for i, e in enumerate(buffer) if e != 0]
               Header['SourceName'] = ''.join(chr(i) for i in buffer[inds])
               
               buffer = np.asarray(struct.unpack('h'*8,file.read(8*2)))
               inds = [i for i, e in enumerate(buffer) if e != 0]
               Header['ImageMode'] = ''.join(chr(i) for i in buffer[inds])                               
               Header['LPFStrength'] = struct.unpack('d',file.read(8))[0]               
               Header['isAutoFlatten'] = struct.unpack('i',file.read(4))[0]        
               Header['isACTrack'] = struct.unpack('i',file.read(4))[0]          
               Header['Width'] = struct.unpack('i',file.read(4))[0]          
               Header['Height'] = struct.unpack('i',file.read(4))[0]        
               Header['Angle'] = struct.unpack('d',file.read(8))[0]          
               Header['IsSineScan'] = struct.unpack('i',file.read(4))[0]          
               Header['OverScan'] = struct.unpack('d',file.read(8))[0]          
               Header['IsFastScanForward'] = struct.unpack('i',file.read(4))[0]          
               Header['IsSlowScanUpward'] = struct.unpack('i',file.read(4))[0]        
               Header['IsXYSwapped'] = struct.unpack('i',file.read(4))[0]          
               Header['XScanSize'] = struct.unpack('d',file.read(8))[0]
               Header['YScanSize'] = struct.unpack('d',file.read(8))[0]
               Header['XOffset'] = struct.unpack('d',file.read(8))[0]
               Header['YOffset'] = struct.unpack('d',file.read(8))[0]
               Header['ScanRate'] = struct.unpack('d',file.read(8))[0]
               Header['SetPoint'] = struct.unpack('d',file.read(8))[0]
               
               buffer = np.asarray(struct.unpack('h'*8,file.read(8*2)))
               inds = [i for i, e in enumerate(buffer) if e != 0]
               Header['SetPointUnit'] = ''.join(chr(i) for i in buffer[inds]) 
               
               Header['TipBias'] = struct.unpack('d',file.read(8))[0]
               Header['SampleBias'] = struct.unpack('d',file.read(8))[0]
               Header['ZGain'] = struct.unpack('d',file.read(8))[0]
               Header['ZScale'] = struct.unpack('d',file.read(8))[0]
               Header['ZOffset'] = struct.unpack('d',file.read(8))[0]
     
               buffer = np.asarray(struct.unpack('h'*8,file.read(8*2)))
               inds = [i for i, e in enumerate(buffer) if e != 0]
               Header['ZUnit'] = ''.join(chr(i) for i in buffer[inds])          
               
               Header['Min'] = struct.unpack('i',file.read(4))[0]
               Header['Max'] = struct.unpack('i',file.read(4))[0]
               Header['nDataAvg'] = struct.unpack('I',file.read(4))[0]
               Header['nCompression'] = struct.unpack('I',file.read(4))[0]
               Header['IsLogScale'] = struct.unpack('I',file.read(4))[0]
               Header['IsSquare'] = struct.unpack('I',file.read(4))[0]
               Header['ZServoGain'] = struct.unpack('d',file.read(8))[0]
               Header['ZScannerRange'] = struct.unpack('d',file.read(8))[0]
               
               
               buffer = np.asarray(struct.unpack('h'*8,file.read(8*2)))
               inds = [i for i, e in enumerate(buffer) if e != 0]
               Header['XYVoltageMode'] = ''.join(chr(i) for i in buffer[inds]) 
    
               buffer = np.asarray(struct.unpack('h'*8,file.read(8*2)))
               inds = [i for i, e in enumerate(buffer) if e != 0]
               Header['ZVoltageMode'] = ''.join(chr(i) for i in buffer[inds])
    
               buffer = np.asarray(struct.unpack('h'*8,file.read(8*2)))
               inds = [i for i, e in enumerate(buffer) if e != 0]
               Header['XYServoMode'] = ''.join(chr(i) for i in buffer[inds])
               
               Header['DataType'] = struct.unpack('I',file.read(4))[0]
               Header['NumXPDDRegion'] = struct.unpack('I',file.read(4))[0]
               Header['NumYPDDRegion'] = struct.unpack('I',file.read(4))[0]
               Header['Amplitude'] = struct.unpack('d',file.read(8))[0]
               Header['SelFrequency'] = struct.unpack('d',file.read(8))[0]
               Header['HeadTiltAngle'] = struct.unpack('d',file.read(8))[0]
       
               buffer = np.asarray(struct.unpack('h'*8,file.read(8*2)))
               inds = [i for i, e in enumerate(buffer) if e != 0]
               Header['szCantilever'] = ''.join(chr(i) for i in buffer[inds])
               
               Header['NCMDrivePercent'] = struct.unpack('d',file.read(8))[0]
               Header['IntensityFactor'] = struct.unpack('d',file.read(8))[0]
    
               # get pdd data [50440]
               PDDRegion = [];
               count = 1
               if 'PSIAPddRegion' in idfs:
                   file.seek(idfs['PSIAPddRegion']['value'],0)
                   for i in range(0,Header['NumXPDDRegion']['value']):
                       PDD = {'Pixel' : struct.unpack('i',file.read(4))[0]}
                       PDD['ScanSize'] = struct.unpack('d',file.read(8))[0]
                       PDD['Offset'] = struct.unpack('d',file.read(8))[0]
                       PDDRegion[count,0] = PDD;
                       count += 1
                       
                   for i in range(0,Header['NumXPDDRegion']['value']):
                       PDD['Pixel'] = struct.unpack('i',file.read(4))[0]
                       PDD['ScanSize'] = struct.unpack('d',file.read(8))[0]
                       PDD['Offset'] = struct.unpack('d',file.read(8))[0]
                       PDDRegion[count,0] = PDD;
                       count += 1
            
               # get zdata [50434]
               file.seek(idfs['PSIAData']['value'],0)
               if Header['DataType']  == 0:
                   ZData = struct.unpack('h'*Header['Width']*Header['Height'],file.read(2*Header['Width']*Header['Height']))
               elif Header['DataType']  == 1:
                   ZData = struct.unpack('i'*Header['Width']*Header['Height'],file.read(4*Header['Width']*Header['Height']))
               elif Header['DataType']  == 2:
                   ZData = struct.unpack('f'*Header['Width']*Header['Height'],file.read(4*Header['Width']*Header['Height']))
               
               
               ZData = np.reshape(np.asarray(ZData),(Header['Height'],Header['Width']));
               ColorMap = ColorMap/np.max(ColorMap)
               
               ImageModel = {'ZData': ZData, 'ColorMap': ColorMap, 'Header' : Header}
                
    return ImageModel











