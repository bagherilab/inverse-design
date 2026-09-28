#!/usr/bin/env python3
"""Reproduce the submitted S7 histogram layout from archived metric CSVs.

Histogram convention recovered from e0ab8a1 analyze_aggregated_results.py:
20 bins per distribution, 1.5-IQR exclusion, alpha=.5, targets in both rows.
Axes/ticks reproduce the submitted composite; no raster content is reused.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inverse_design.utils.utils import remove_outliers
METRICS=['doub_time','doub_time_std','symmetry','symmetry_std','colony_growth']
LABELS=['Doubling time (hr)','Doubling time std (hr)','Symmetry','Symmetry std','Colony growth (µm/day)']
COLORS=['#bb883b','#ddc39d','#486b45','#679A63','#545aab']
# Pixel landmarks of the existing composite, used solely to specify vector axes.
BASELINES=[[269,270,272,277,285],[535,537,540,535,537]]
TOPS=[[68,68,71,76,83],[335,336,339,334,335]]
LEFTS=[[490,1343,2182,3042,3894],[500,1344,2182,3043,3894]]
RIGHTS=[[1264,2116,2957,3814,4667],[1274,2116,2958,3814,4667]]
TICKS=[[[688,946,1203],[1407,1672,1937],[2319.5,2605,2891],[3107,3313,3518.5,3725],[4047,4255.5,4462]],[[698,956,1213],[1408,1673,1937.5],[2319.5,2606,2892],[3108,3313.5,3519,3725.5],[4047,4255.5,4462]]]
VALUES=[[50,100,150],[0,10,20],[.6,.8,1],[0,.05,.1,.15],[0,25,50]]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--legacy-labels',action='store_true');a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 target=json.loads((a.base_dir/'targets.json').read_text());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1,'xtick.major.width':1.5,'ytick.major.width':1.5})
 fig=plt.figure(figsize=(4702/300,752/300),dpi=300,facecolor='white');data=[]
 for row,g in enumerate([0,4]):
  df=pd.read_csv(a.base_dir/f'iter_{g}/final_metrics.csv')
  for col,m in enumerate(METRICS):
   left,right=LEFTS[row][col],RIGHTS[row][col];bottom,top=BASELINES[row][col],TOPS[row][col]
   ax=fig.add_axes([left/4702,(752-bottom)/752,(right-left)/4702,(bottom-top)/752]);v=df[m].replace([np.inf,-np.inf],np.nan).dropna();v,_=remove_outliers(v,1.5)
   counts,edges=np.histogram(v,bins=20);color='gray' if row==0 else tuple(np.array(matplotlib.colors.to_rgb(COLORS[col]))*.9)
   ax.bar(edges[:-1],counts,np.diff(edges),align='edge',color=(*matplotlib.colors.to_rgb(color),.5),edgecolor='black',linewidth=1)
   tr=np.polyfit(VALUES[col],TICKS[row][col],1);ax.set_xlim((left-tr[1])/tr[0],(right-tr[1])/tr[0]);ax.set_ylim(0,250);ax.set_xticks(VALUES[col]);ax.set_yticks([0,250]);ax.grid(axis='y',ls='--',color='.82');ax.set_axisbelow(True)
   ax.axvline(target[m],color='red',ls='--',alpha=.8,lw=2);ax.spines[['top','right']].set_visible(False)
   if row==0:ax.tick_params(labelbottom=False)
   else:
    ax.set_xlabel(LABELS[col],fontweight='bold',labelpad=7)
    if col in (2,3):ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}' if not a.legacy_labels else f'{x:g}'))
   if col:ax.tick_params(labelleft=False)
   for lab in ax.get_xticklabels()+ax.get_yticklabels():lab.set_fontweight('bold')
   data.append(dict(generation=g,metric=m,counts=counts.tolist(),edges=edges.tolist(),target=target[m]))
 fig.text(.043,.775,'g0' if a.legacy_labels else 'gen-0',fontsize=12,fontweight='bold',ha='center');fig.text(.043,.38,'g4' if a.legacy_labels else 'gen-4',fontsize=12,fontweight='bold',ha='center')
 fig.text(.068,.59,'Number of samples',rotation=90,va='center',ha='center',fontsize=12,fontweight='bold')
 # Preserve the original generation-axis title.
 fig.text(.009,.58,'ABC generation',rotation=90,va='center',ha='center',fontsize=12,fontweight='bold')
 fig.add_artist(plt.Line2D([107/4702,107/4702],[45/752,710/752],transform=fig.transFigure,color='black',linewidth=2))
 fig.savefig(a.output_dir/'A_histograms.png',dpi=300);fig.savefig(a.output_dir/'A_histograms.pdf');(a.output_dir/'plotted_values.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
