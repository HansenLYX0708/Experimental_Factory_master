# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:42:26 2023

@author: 1000145667
"""

"""
Park AFM PTR Automated Metrology script
auto-anomaly detection for HAMR heads
by Guilherme "Will" Souza for Western Digital
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import filedialog
import os
import glob
import pandas as pd
import sys
from scipy.optimize import curve_fit
import itertools
from matplotlib import cm
import ParkSystemsTiffReader as ParkSystemsTiffReader
import cv2
import math
import scipy.ndimage
from PIL import Image


def planeFit1(image,xa):
    # ---- 1st order plane fit (Z = aX + bY + c) at ROI only ----
    
    x = np.arange(xa, 1024, 1)
    y = np.arange(256)
    
    Z = image[:,xa:1024]
    
    X, Y = np.meshgrid(x, y)
    x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()
    
    def func3(xy, a, b, c):
        x, y = xy
        return a*x + b*y + c
    
    popt3, pcov3 = curve_fit(func3, (x1, y1), z1)
    
    # z_ = popt3[0]*X + popt3[1]*Y + popt3[2]
    
    # Applying for the whole scan
    x_all = np.arange(1024)
    y_all = np.arange(256)
    
    X_all, Y_all = np.meshgrid(x_all, y_all)
    
    z__ = popt3[0]*X_all + popt3[1]*Y_all + popt3[2] # Z = aX + bY + c; X and Y from meshgrid
    
    image_final = image - z__
    
    return image_final





def Will_scratch_finder(image):
    
    Will_scratch = np.zeros((np.size(image,0),np.size(image,1)))
    
    for i in range(0,np.size(image,1)):
        
        #print(i)
        
        if i == 0:                
            line_ = np.average(image[:,0:1], axis=1)
        elif i == 1:                
            line_ = np.average(image[:,1:2], axis=1)
        elif i == np.size(image,1) - 1:                
            line_ = np.average(image[:,np.size(image,1) - 2:np.size(image,1) - 1], axis=1)
        elif i == np.size(image,1):
            line_ = np.average(image[:,np.size(image,1) - 1:np.size(image,1)], axis=1)
        else:
            line_ = np.average(image[:,i-1:i+1], axis=1)
            
        
        line_mean = np.mean(line_)
        line_STD = np.std(line_)
        scratch_thres = line_mean - (2.00 * line_STD)
        #line_[line_ >= scratch_thres] = 0
        #line_[line_ < scratch_thres] = 1
        for i2 in range(0,len(line_)):
            if line_[i2] < scratch_thres:
                line_[i2] = 1
            else:
                line_[i2] = 0

        
        Will_scratch[:,i] = line_
    
    
    return Will_scratch
    
    


# Dialog window
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
directory = filedialog.askdirectory(initialdir='C:/', title='Please select folder with *.TIFF Park PTR (PMR) files:')
os.chdir(directory)
filenames = glob.glob('*.TIFF')
if not os.path.exists("Will-PTR-Park-HAMR-Anomaly"):
    os.mkdir('Will-PTR-Park-HAMR-Anomaly')



block1_abvABS_quant_norm = list(itertools.repeat('FAILED', len(filenames)))
block2_abvABS_quant_norm = list(itertools.repeat('FAILED', len(filenames)))
block3_abvABS_quant_norm = list(itertools.repeat('FAILED', len(filenames)))
block1_abvABSper = list(itertools.repeat('FAILED', len(filenames)))
block2_abvABSper = list(itertools.repeat('FAILED', len(filenames)))
block3_abvABSper = list(itertools.repeat('FAILED', len(filenames)))
P1S2_max = list(itertools.repeat('FAILED', len(filenames)))
block1_Wscr_percent = list(itertools.repeat('FAILED', len(filenames)))
block2_Wscr_percent = list(itertools.repeat('FAILED', len(filenames)))
block3_Wscr_percent = list(itertools.repeat('FAILED', len(filenames)))




for j in range(0,len(filenames)):
    
    try:
        
        
        A = ParkSystemsTiffReader.ParkSystemsTiffReader(filenames[j])
        matrix_original = np.array(A['ZData']) * A['Header']['ZGain']
        
        
        print('------------------------------')
        print(filenames[j])
        print('Expected config: 1024 x 256 pixels; 40 x 20 um')
        print(str(A['Header']['Height']) + ' x ' + str(A['Header']['Width']) + ' pixels')
        print(str(A['Header']['YScanSize']) + ' x ' + str(A['Header']['XScanSize']) + ' um')
        print('------------------------------')
        
        
        matrix_original = cv2.resize(matrix_original, (256,1024)) # forced PHO 128 --> 256
        matrix_original = np.flipud(matrix_original)
        
        
        # Create a re-scaled version of the original scan for plotting purposes only
        temp0 = np.max(matrix_original) - np.min(matrix_original)
        temp1 = 8 / temp0
        temp2 = matrix_original * temp1
        temp3 = np.min(temp2) - (-4)
        data00 = temp2 - temp3
        #data00_ = cv2.resize(data00, (128,1024))
        data00__ = cv2.rotate(data00, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        
        line_prof_raw_00 = np.average(matrix_original[0:1024,0:256], axis=1)
        line_prof_raw_sm_00 = savgol_filter(line_prof_raw_00, 21, 2)
        line_prof_raw_sm_der_00 = np.gradient(line_prof_raw_sm_00)
        line_prof_raw_sm_der_00[line_prof_raw_sm_der_00 < 0] = 0
        #LL = np.where(line_prof_raw_sm_der_00[300::] == max(line_prof_raw_sm_der_00[300::]))[0][0] + 300 + 20
        
        #lineDerMaxLocal = max(line_prof_raw_sm_der_00[650:1000])
        peaksLineDer = find_peaks(line_prof_raw_sm_der_00[650:1000], distance=12)[0] + 650
        
        #contingency:
        if len(peaksLineDer) < 7:
            
            line_prof_raw_00 = np.average(matrix_original[0:1024,118:138], axis=1)
            line_prof_raw_sm_00 = savgol_filter(line_prof_raw_00, 7, 2)
            line_prof_raw_sm_der_00 = np.gradient(line_prof_raw_sm_00)
            line_prof_raw_sm_der_00[line_prof_raw_sm_der_00 < 0] = 0
            #LL = np.where(line_prof_raw_sm_der_00[300::] == max(line_prof_raw_sm_der_00[300::]))[0][0] + 300 + 20
            
            #lineDerMaxLocal = max(line_prof_raw_sm_der_00[650:1000])
            peaksLineDer = find_peaks(line_prof_raw_sm_der_00[650:1000], distance=12)[0] + 650
            
            
        
        peaksLineDer_y = []
        for p1 in peaksLineDer:
            peaksLineDer_y.append(line_prof_raw_sm_der_00[p1])
        
        sorted_indices5 = np.argsort(peaksLineDer_y)[::-1]
        peaksLineDer_y_ranked = [peaksLineDer_y[i] for i in sorted_indices5]
        #peaks5_ranked = [peaksLineDer[i] for i in sorted_indices5]
        lineDerMaxLocal = np.median(peaksLineDer_y_ranked[0:7])
        
        for k in range(1000,300,-1):
            #print(str(k))
            if line_prof_raw_sm_der_00[k] > 3.5 * lineDerMaxLocal:
                break
        
        LL = k + 20
        
        
        # Line_by_Line 1st order Leveling Loop (high res direction) using mean of AlTiC
        matrix_all2 = np.zeros((1024, 256))
        
        
        def func1(x, a, b):
            return a + b*x
        
        x_ = np.linspace(LL, 1024, 1024-LL+1)
        x__ = np.linspace(1, 1024, 1024)
        
        for n in range(0,256):
            #y_ = savgol_filter(matrix_original[LL-1::,n], 50, 2)
            if n == 0:                
                y_ = np.average(matrix_original[LL-1::,0:2], axis=1)
            elif n == 1:                
                y_ = np.average(matrix_original[LL-1::,0:3], axis=1)
            elif n == 254:                
                y_ = np.average(matrix_original[LL-1::,252:255], axis=1)
            elif n == 255:                
                y_ = np.average(matrix_original[LL-1::,253:255], axis=1)
            else:
                y_ = np.average(matrix_original[LL-1::,n-2:n+2], axis=1)
            
            y_ = savgol_filter(y_, 21, 2)
            popt1, pcov1 = curve_fit(func1, x_, y_)
            matrix_all2[0:1024, n] = matrix_original[0:1024,n] - func1(x__, *popt1)
            

        
        
        #data0_ = cv2.resize(matrix_all2, (128,1024))
        data0 = cv2.rotate(matrix_all2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        
       
        range_R = LL
        range_L = np.where(line_prof_raw_sm_der_00[50:k - 190] == max(line_prof_raw_sm_der_00[50:k - 190]))[0][0] + 50 - 20
            
        
        
        data2_fp = planeFit1(data0,range_R) * 1000 # micron-to-nanometer
        #data2_fp = data0 * 1000 # micron-to-nanometer
        
        
        ##### Zero the ABS based on filtered histogram
        try:
            
            ShL = data2_fp[32:224,range_R::]
            ShL_sizey = len(ShL); ShL_sizex = len(ShL[0])
            ShL_reshape = np.reshape(ShL,(ShL_sizex*ShL_sizey,1))
            ShL_hist = plt.hist(ShL_reshape, bins='auto')
            plt.close()
            ShL_hist_x = np.linspace(min(ShL_hist[1]), max(ShL_hist[1]), len(ShL_hist[0]))
            ShL_hist_sm = savgol_filter(ShL_hist[0], 9, 2)
            ShL_hist_sm_max = max(ShL_hist_sm)
            peaksShL, _ = find_peaks(ShL_hist_sm, height=0.35*ShL_hist_sm_max, distance=20)
            peaksShL_dist = round(2.0 * np.std(ShL_reshape) / (ShL_hist_x[1] - ShL_hist_x[0])) # 2x STD
            #peaksShL_dist = round(0.63*abs(peaksShL[0] - peaksShL[1]))
            if peaksShL[0] < peaksShL[1]:
                hist_cut_L = peaksShL[0] - peaksShL_dist
                hist_cut_R = peaksShL[1] + peaksShL_dist
            else:
                hist_cut_L = peaksShL[1] - peaksShL_dist
                hist_cut_R = peaksShL[0] + peaksShL_dist
            
            if hist_cut_L > 0 and hist_cut_L < len(ShL_hist_x):
                ShL_reshape[ShL_reshape < ShL_hist_x[hist_cut_L]] = np.nan
            if hist_cut_R > 0 and hist_cut_R < len(ShL_hist_x):
                ShL_reshape[ShL_reshape > ShL_hist_x[hist_cut_R]] = np.nan
            
            
            data2_fp = data2_fp - np.nanmean(ShL_reshape)
        
        except Exception as e:
            print('')
            print('------------------------------------------')
            print('*** No histogram-filtered ABS possible ***')
            print('------------------------------------------')
            print('')
            
        #####
        
        # Deglitching
        data2_fp[data2_fp < -5.0] = -5.0
        data2_fp[data2_fp > 3.0] = 3.0
        
        # Rotation correction
        data2_fp_original = data2_fp.copy()
        
        Y1_ = np.average(data2_fp[6:14,range_R-35:range_R+15], axis=0)
        Y1_sm = savgol_filter(Y1_, 11, 3)
        Y1_sm_der = np.gradient(Y1_sm)
        Y1_sm_der_max = np.where(Y1_sm_der == max(Y1_sm_der))[0][0] + range_R-35
        
        Y2_ = np.average(data2_fp[242:250,range_R-35:range_R+15], axis=0)
        Y2_sm = savgol_filter(Y2_, 11, 3)
        Y2_sm_der = np.gradient(Y2_sm)
        Y2_sm_der_max = np.where(Y2_sm_der == max(Y2_sm_der))[0][0] + range_R-35
        
        data2_fp = cv2.resize(data2_fp, (1024,1024))
        m = (Y2_sm_der_max - Y1_sm_der_max)/(4*(246 - 10))
        angle = math.atan(m)
        angle_corr = math.cos(angle)
        data2_fp = scipy.ndimage.rotate(data2_fp,-math.degrees(angle),reshape=False,mode='constant',cval=0)
        data2_fp = cv2.resize(data2_fp, (1024,256))

        # save data00 data2_fp
        im_flattened_data = Image.fromarray(data2_fp)
        im_flattened_data.save('Will-PTR-Park-HAMR-Anomaly/' + filenames[j][0:-4] + '_will.tiff')
        im_flattened_data = Image.fromarray(data00__)
        im_flattened_data.save('Will-PTR-Park-HAMR-Anomaly/' + filenames[j][0:-4] + '_data00.tiff')



        #line_prof_2fp = np.average(data2_fp[0:256,0:1024], axis=0)
        line_prof_2fp = np.average(data2_fp[32:224,0:1024], axis=0)
        
        #line_prof_raw00 = np.average(data00__[0:256,0:1024], axis=0)
        line_prof_raw00 = np.average(data00__[32:224,0:1024], axis=0)    
        

        # Plotting 2D
        lpr00_min = min(line_prof_raw00)
        lpr00_max = max(line_prof_raw00)
        lpr2fp_min = min(line_prof_2fp)
        lpr2fp_max = max(line_prof_2fp)
        line_prof_raw00 = line_prof_raw00 * ((lpr2fp_max - lpr2fp_min)/(lpr00_max - lpr00_min))
        lpr00_min = min(line_prof_raw00)
        line_prof_raw00 = line_prof_raw00 - (lpr00_min - lpr2fp_min)
        
        fig_02, ax = plt.subplots(figsize =(15, 5), layout="constrained")
        ax.plot(line_prof_raw00, '-b', lw=1.5, label="Park Raw Scan")
        ax.plot(line_prof_2fp, '-r', lw=1.5, label="Will's Transform")
        ax.set_title("THO MSL AFM Park Auto-PTR", weight='bold', fontsize=18)
        ax.legend(loc='lower right', fontsize=13)
        ax.set_xlim([0, 1024])
        ax.axhline(y=0, color='k', linestyle='--',lw=0.8)
        plt.savefig('Will-PTR-Park-HAMR-Anomaly/' + filenames[j][0:-4] + '_fig01.jpg', bbox_inches = 'tight', pad_inches = 0,dpi=150)
        plt.show()
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), layout="constrained")
        fig.suptitle('THO MSL AFM Park Auto-PTR', weight='bold', fontsize=18)
        ax1.imshow(data00__, cmap=cm.afmhot)
        ax1.axis('off')
        ax1.set_aspect('auto')
        ax1.set_title('Park Raw Scan', fontsize=16)
        ax2.imshow(data2_fp, cmap=cm.afmhot)
        ax2.axis('off')
        ax2.set_aspect('auto')
        ax2.set_title("Will's Transform", fontsize=16)
        plt.savefig('Will-PTR-Park-HAMR-Anomaly/' + filenames[j][0:-4] + '_fig02.jpg', bbox_inches = 'tight', pad_inches = 0,dpi=150)
        plt.show()
        
        
        
        # Plotting 3D
        x = np.arange(1024)
        y = np.arange(256)
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize =(15, 8), layout="constrained")
        fig.suptitle('THO MSL AFM Park Auto-PTR', weight='bold', fontsize=18)
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(X, Y, data00__, cmap=cm.afmhot, antialiased=False)
        ax.azim = -105
        ax.elev = 20
        ax.text2D(0.05, 0.95, "Park Raw Scan", transform=ax.transAxes, size=16)
        ax.set_xlabel('x [pixels]')
        ax.set_ylabel('y [pixels]')
        ax.set_zlabel('z [nm]')
        ax.set_zlim([-15,15])
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_surface(X, Y, data2_fp_original, cmap=cm.afmhot, antialiased=False)
        ax.azim = -105 #75
        ax.elev = 20 #20
        ax.text2D(0.05, 0.95, "Will's Transform", transform=ax.transAxes, size=16)
        ax.set_xlabel('x [pixels]')
        ax.set_ylabel('y [pixels]')
        ax.set_zlabel('z [nm]')
        ax.set_zlim([-15,15]) #10,10
        plt.savefig('Will-PTR-Park-HAMR-Anomaly/' + filenames[j][0:-4] + '_fig03.jpg', bbox_inches = 'tight', pad_inches = 0,dpi=150)
        plt.show()

    
        
        
        # Blocks and Metrics                
        
        block01 = data2_fp[10:246,k - 395:k - 285]
        block02 = data2_fp[10:246,k - 255:k - 175]
        block03 = data2_fp[10:246,k - 155:k - 108]
        
       
        
        
        block1_abvABS = block01.copy()
        block1_abvABS[block1_abvABS < 0] = np.nan
        block1_abvABS_quant_norm[j] = round(np.nansum(block1_abvABS) / block1_abvABS.size, 2)
        block1_abvABSper_ = block1_abvABS.copy()
        block1_abvABSper_[block1_abvABSper_ >= 0] = 1
        block1_abvABSper[j] = round((np.nansum(block1_abvABSper_) / block1_abvABSper_.size) * 100, 2)
        block1_abvABSper_[block1_abvABSper_ != 1] = 0
        
        block2_abvABS = block02.copy()
        block2_abvABS[block2_abvABS < 0] = np.nan
        block2_abvABS_quant_norm[j] = round(np.nansum(block2_abvABS) / block2_abvABS.size, 2)
        block2_abvABSper_ = block2_abvABS.copy()
        block2_abvABSper_[block2_abvABSper_ >= 0] = 1
        block2_abvABSper[j] = round((np.nansum(block2_abvABSper_) / block2_abvABSper_.size) * 100, 2)
        block2_abvABSper_[block2_abvABSper_ != 1] = 0
        
        block3_abvABS = block03.copy()
        block3_abvABS[block3_abvABS < 0] = np.nan
        block3_abvABS_quant_norm[j] = round(np.nansum(block3_abvABS) / block3_abvABS.size, 2)
        block3_abvABSper_ = block3_abvABS.copy()
        block3_abvABSper_[block3_abvABSper_ >= 0] = 1
        block3_abvABSper[j] = round((np.nansum(block3_abvABSper_) / block3_abvABSper_.size) * 100, 2)
        block3_abvABSper_[block3_abvABSper_ != 1] = 0
        

        
        
        
        fig, (ax1, ax2, ax3,ax4, ax5, ax6,ax7, ax8, ax9) = plt.subplots(1, 9, figsize=(23.75, 9.5), layout="constrained")
        fig.suptitle('Shield above ABS auto-detector [Will]', weight='bold', fontsize=18)
        ax1.imshow(block01, cmap=cm.afmhot)
        ax1.axis('off')
        ax1.set_aspect('auto')
        ax1.set_title('Block1', fontsize=16)
        ax2.imshow(block1_abvABSper_, cmap=cm.gist_gray)
        ax2.axis('off')
        ax2.set_aspect('auto')
        ax2.set_title("Above ABS = " + str(block1_abvABSper[j]) + "%", fontsize=16)
        ax3.imshow(block1_abvABS, cmap=cm.hsv)
        ax3.axis('off')
        ax3.set_aspect('auto')
        ax3.set_title("Above ABS = " + str(block1_abvABS_quant_norm[j]), fontsize=16)
        ax4.imshow(block02, cmap=cm.afmhot)
        ax4.axis('off')
        ax4.set_aspect('auto')
        ax4.set_title('Block2', fontsize=16)
        ax5.imshow(block2_abvABSper_, cmap=cm.gist_gray)
        ax5.axis('off')
        ax5.set_aspect('auto')
        ax5.set_title("Above ABS = " + str(block2_abvABSper[j]) + "%", fontsize=16)
        ax6.imshow(block2_abvABS, cmap=cm.hsv)
        ax6.axis('off')
        ax6.set_aspect('auto')
        ax6.set_title("Above ABS = " + str(block2_abvABS_quant_norm[j]), fontsize=16)
        ax7.imshow(block03, cmap=cm.afmhot)
        ax7.axis('off')
        ax7.set_aspect('auto')
        ax7.set_title('Block3', fontsize=16)
        ax8.imshow(block3_abvABSper_, cmap=cm.gist_gray)
        ax8.axis('off')
        ax8.set_aspect('auto')
        ax8.set_title("Above ABS = " + str(block3_abvABSper[j]) + "%", fontsize=16)
        ax9.imshow(block3_abvABS, cmap=cm.hsv)
        ax9.axis('off')
        ax9.set_aspect('auto')
        ax9.set_title("Above ABS = " + str(block3_abvABS_quant_norm[j]), fontsize=16)
        plt.savefig('Will-PTR-Park-HAMR-Anomaly/' + filenames[j][0:-4] + '_fig04.jpg', bbox_inches = 'tight', pad_inches = 0,dpi=150)
        plt.show()
        
        
        
        P1S2 = data2_fp[10:246,k - 275:k - 245]
        P1S2_ = np.average(P1S2, axis=0)
        P1S2_max[j] = round(max(P1S2_), 2)
        P1S2_max_x = np.where(P1S2_ == max(P1S2_))[0][0] + k - 275
        
        
        line_prof_2fp_final = np.average(data2_fp[10:246,0:1024], axis=0)
        x_axis = np.linspace(0, 40, 1024)
        
        
        fig_03 = plt.figure(figsize =(15, 5), layout="constrained")
        plt.plot(x_axis, line_prof_2fp_final, '-g', lw=1.5)
        plt.title("uPTR profile", weight='bold', fontsize=18)
        plt.xlim([0, 40])
        plt.axhline(y=0, color='k', linestyle='--',lw=0.65)
        plt.axvline(x=x_axis[P1S2_max_x], color='r', linestyle='--',lw=1.75)
        plt.xlabel("[um]")
        plt.ylabel("Topography [nm]")
        plt.savefig('Will-PTR-Park-HAMR-Anomaly/' + filenames[j][0:-4] + '_fig05.jpg', bbox_inches = 'tight', pad_inches = 0,dpi=150)
        plt.show()
        
        
        
        
        block01_Will_scratch = Will_scratch_finder(block01)
        block02_Will_scratch = Will_scratch_finder(block02)
        block03_Will_scratch = Will_scratch_finder(block03)
    
        block1_Wscr_per = round((np.sum(block01_Will_scratch) / block01.size) * 100, 2)
        block2_Wscr_per = round((np.sum(block02_Will_scratch) / block02.size) * 100, 2)
        block3_Wscr_per = round((np.sum(block03_Will_scratch) / block03.size) * 100, 2)
        
        block1_Wscr_percent[j] = block1_Wscr_per
        block2_Wscr_percent[j] = block2_Wscr_per
        block3_Wscr_percent[j] = block3_Wscr_per
        
                
        
        
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(15, 9), layout="constrained")
        fig.suptitle('Scratch/local cavity auto-detector [Will]', weight='bold', fontsize=18)
        ax1.imshow(block01, cmap=cm.afmhot)
        ax1.axis('off')
        ax1.set_aspect('auto')
        ax1.set_title('Block1', fontsize=16)
        ax2.imshow(block01_Will_scratch, cmap=cm.gist_gray)
        ax2.axis('off')
        ax2.set_aspect('auto')
        ax2.set_title("Anomaly = " + str(block1_Wscr_per) + "%", fontsize=16)
        ax3.imshow(block02, cmap=cm.afmhot)
        ax3.axis('off')
        ax3.set_aspect('auto')
        ax3.set_title('Block2', fontsize=16)
        ax4.imshow(block02_Will_scratch, cmap=cm.gist_gray)
        ax4.axis('off')
        ax4.set_aspect('auto')
        ax4.set_title("Anomaly = " + str(block2_Wscr_per) + "%", fontsize=16)
        ax5.imshow(block03, cmap=cm.afmhot)
        ax5.axis('off')
        ax5.set_aspect('auto')
        ax5.set_title('Block3', fontsize=16)
        ax6.imshow(block03_Will_scratch, cmap=cm.gist_gray)
        ax6.axis('off')
        ax6.set_aspect('auto')
        ax6.set_title("Anomaly = " + str(block3_Wscr_per) + "%", fontsize=16)
        plt.savefig('Will-PTR-Park-HAMR-Anomaly/' + filenames[j][0:-4] + '_fig06.jpg', bbox_inches = 'tight', pad_inches = 0, dpi=150)
        plt.show()
            
            
        
        
        
    except Exception as e:
        print("An error occurred:", e)
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        print(filenames[j])
        

# Create a Pandas Dataframe for exporting into Excel
df = pd.DataFrame()
df['Data location'] = [directory] * len(filenames)
df['Filename'] = filenames
df['Block1 Anomaly_Will [%]'] = block1_Wscr_percent
df['Block2 Anomaly_Will [%]'] = block2_Wscr_percent
df['Block3 Anomaly_Will [%]'] = block3_Wscr_percent
df['Block1 Above ABS [%]'] = block1_abvABSper
df['Block2 Above ABS [%]'] = block2_abvABSper
df['Block3 Above ABS [%]'] = block3_abvABSper
df['Block1 Above ABS [normalized]'] = block1_abvABS_quant_norm
df['Block2 Above ABS [normalized]'] = block2_abvABS_quant_norm
df['Block3 Above ABS [normalized]'] = block3_abvABS_quant_norm
df['P1S2 Protrusion [nm]'] = P1S2_max



# Write DF to Excel
writer = pd.ExcelWriter('Will-PTR-Park-HAMR-Anomaly/Will-PTR-Park-HAMR-Anomaly.xlsx')
df.to_excel(writer,'Sheet1', index=False)
writer._save()