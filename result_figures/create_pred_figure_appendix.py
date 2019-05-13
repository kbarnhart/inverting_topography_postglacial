
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:01:42 2018

@author: barnhark
"""
import glob
import os
import numpy as np


fin = np.sort(glob.glob(os.path.join(os.path.sep, *['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_figures', 'prediction', 'sew' , 'BEST_PARAMETERS', 'uncertainty_comparison.sew.*.pdf'])))

fOut=r'prediction_appendix_figures.tex'
folder_name = 'prediction_figures'
#%%
f = open(fOut, 'w')
#%%
sz='5'
for fname in fin:

    fig_name = os.path.split(fname)[-1].replace('.', '_', 2)
    filepath = folder_name + '/' + fig_name

    location = fig_name.split('_')[-1].split('.')[0]
    
    print(location)


    f.write('% prediction summary output for location ' + location +'\n')
    f.write('\clearpage \n')
    f.write('\\begin{figure}% \n')
    f.write('\\includegraphics[width=6.5in]{' + filepath + '}\n')


    f.write(('        \caption{Summary of prediction results at ' + location + ' showing expected elevation and uncertainty through time. '
             'The gray box is a 50 foot deep reference box that extends below the modern surface. Three expected values and 95\% confidence '
             'regions are shown that corresponds to the two approaches to model selection (only 842 and all nine 800 variants) and the two '
             'approaches to model structure and calibration uncertainty (independent or covarying).}% \n'))
    f.write('   \label{fig:pred_summary_' + location + '}% \n')
    f.write(' \end{figure}\n')
    f.write(' \n')
    f.write(' \n')





f.close()

