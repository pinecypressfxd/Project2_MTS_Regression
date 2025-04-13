## Predicting the characteristics of bursty bulk flows in the Earth’s plasma sheet using machine learning techniques 

  - XGBoost_regression_code: model training with different parameter combination 
  - TM_03_model: model training with TM-03 model prediction as additional background


Figure6: Note that we directly use the ML prediction results of the maximum values as “Max” in the table. To calculate the “Min” for the plotting, if the MAPE value of the “Range” is lower than the original MAPE of the “Min” value, we use a proxy for the minimum value, calculated as the maximum value minus the range value. Otherwise, we use the originally predicted minimum value directly to calculate the MAPE of the “Min”. For B_z, |B|, P_m, |V_i |, |V_(i⊥) |, and P_p, the minimum values are calculated indirectly.![image](https://github.com/user-attachments/assets/89636c00-6bc8-43ed-bbdf-1d2a7eedfd3c)

