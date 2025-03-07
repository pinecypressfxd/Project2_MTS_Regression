import numpy as np
import pandas as pd
def Tps_tm2003_new(xgsm, ygsm, n_sw, v_sw, imf_bz):
    # Coefficients
    A1 = np.array([1.678, -0.159])
    A2 = np.array([-0.1606, 0.608])
    A3 = np.array([1.669, 0.5055])
    A4 = np.array([4.820, 0.0796])
    A5 = np.array([2.855, 0.2746])
    A6 = np.array([-0.602, 0.0361])
    A7 = np.array([-0.836, -0.0342])
    A8 = np.array([-2.491, -0.7935])
    A9 = np.array([0.2568, 1.162])
    A10 = np.array([0.2249, 0.4756])
    A11 = np.array([0.1887, 0.7117])
    A12 = np.array([-0.4458, 0.0])
    A13 = np.array([-0.0331, 0.0])
    A14 = np.array([-0.0241, 0.0])
    A15 = np.array([-2.689, 0.0])
    A16 = np.array([1.222, 0.0])

    # Calculations
    rho = np.sqrt(xgsm**2 + ygsm**2)
    rho_s = rho / 10.0
    phi = -np.arctan2(ygsm, xgsm)
    n_sw_s = n_sw / 10.0
    v_sw_s = v_sw / 500.0

    imf_bs_s, imf_bn_s = (0.0, 0.0)
    if imf_bz < 0.0:
        imf_bs_s = -imf_bz
    else:
        imf_bn_s = imf_bz
    imf_bn_s /= 5.0
    imf_bs_s /= 5.0

    t_ps = (A1[0]*v_sw_s + A2[0]*imf_bn_s + A3[0]*imf_bs_s +
            A4[0]*np.exp(-(A9[0]*v_sw_s**A15[0] + A10[0]*imf_bn_s + A11[0]*imf_bs_s)*(rho_s-1.0)) +
            (A5[0]*v_sw_s + A6[0]*imf_bn_s + A7[0]*imf_bs_s +
             A8[0]*np.exp(-(A12[0]*v_sw_s**A16[0] + A13[0]*imf_bn_s + A14[0]*imf_bs_s)*(rho_s-1.0))) * np.sin(phi)**2)
    
    Tps_tm2003_new = t_ps * 1000.0
    return Tps_tm2003_new

def Nps_tm2003_new(xgsm, ygsm, n_sw, v_sw, imf_bz):
    # Coefficients
    A1 = np.array([1.678, -0.159])
    A2 = np.array([-0.1606, 0.608])
    A3 = np.array([1.669, 0.5055])
    A4 = np.array([4.820, 0.0796])
    A5 = np.array([2.855, 0.2746])
    A6 = np.array([-0.602, 0.0361])
    A7 = np.array([-0.836, -0.0342])
    A8 = np.array([-2.491, -0.7935])
    A9 = np.array([0.2568, 1.162])
    A10 = np.array([0.2249, 0.4756])
    A11 = np.array([0.1887, 0.7117])
    A12 = np.array([-0.4458, 0.0])
    A13 = np.array([-0.0331, 0.0])
    A14 = np.array([-0.0241, 0.0])
    A15 = np.array([-2.689, 0.0])
    A16 = np.array([1.222, 0.0])

    # Calculations
    rho = np.sqrt(xgsm**2 + ygsm**2)
    rho_s = rho / 10.0
    phi = -np.arctan2(ygsm, xgsm)
    n_sw_s = n_sw / 10.0
    v_sw_s = v_sw / 500.0

    imf_bs_s, imf_bn_s = (0.0, 0.0)
    if imf_bz < 0.0:
        imf_bs_s = -imf_bz
    else:
        imf_bn_s = imf_bz
    imf_bn_s /= 5.0
    imf_bs_s /= 5.0

    n_ps = (A1[1] + A2[1]*n_sw_s**A10[1] + A3[1]*imf_bn_s + A4[1]*v_sw_s*imf_bs_s)*rho_s**A8[1]+\
    (A5[1]*n_sw_s**A11[1] + A6[1]*imf_bn_s + A7[1]*v_sw_s*imf_bs_s)*(rho_s**A9[1]) * np.sin(phi)**2
    
    Nps_tm2003_new = n_ps
    return Nps_tm2003_new

def Pps_tm2003_new(xgsm, ygsm, n_sw, v_sw, imf_bz):
    # Coefficients
    A1 = np.array([1.678, -0.159, 0.057])
    A2 = np.array([-0.1606, 0.608, 0.524])
    A3 = np.array([1.669, 0.5055, 0.0908])
    A4 = np.array([4.820, 0.0796, 0.527])
    A5 = np.array([2.855, 0.2746, 0.078])
    A6 = np.array([-0.602, 0.0361, -4.422])
    A7 = np.array([-0.836, -0.0342, -1.533])
    A8 = np.array([-2.491, -0.7935, -1.217])
    A9 = np.array([0.2568, 1.162, 2.54])
    A10 = np.array([0.2249, 0.4756, 0.32])
    A11 = np.array([0.1887, 0.7117, 0.754])
    A12 = np.array([-0.4458, 0.0, 1.048])
    A13 = np.array([-0.0331, 0.0, -0.074])
    A14 = np.array([-0.0241, 0.0, 1.015])
    A15 = np.array([-2.689, 0.0, 0.0])
    A16 = np.array([1.222, 0.0, 0.0])

    # Calculations
    rho = np.sqrt(xgsm**2 + ygsm**2)
    rho_s = rho / 10.0
    phi = -np.arctan2(ygsm, xgsm)
    p_sw = 1.0E-6 * (1.67 + 1.67 * 4 * 0.04) * n_sw * v_sw**2
    p_sw_s = p_sw / 3.0
    n_sw_s = n_sw / 10.0
    v_sw_s = v_sw / 500.0

    imf_bs_s, imf_bn_s, imf_clock = (0.0, 0.0, 0.0)
    if imf_bz < 0.0:
        imf_bs_s = -imf_bz
        imf_clock = np.pi
    else:
        imf_bn_s = imf_bz
    imf_bn_s /= 5.0
    imf_bs_s /= 5.0
    imf_b_perp = abs(imf_bz)
    F_s = imf_b_perp * np.sqrt(np.sin(0.5 * imf_clock)) / 5.0

    Pps_tm2003_new = (A1[2]*rho_s**A6[2] + A2[2]*(p_sw_s**A11[2])*(rho_s**A7[2]) +
                      A3[2]*(F_s**A12[2])*(rho_s**A8[2]) +
                      (A4[2]*(p_sw_s**A13[2])*np.exp(-A9[2]*rho_s) +
                       A5[2]*(F_s**A14[2])*np.exp(-A10[2]*rho_s)) * (np.sin(phi)**2))

    return Pps_tm2003_new


if __name__ == "__main__":
    file_path = '/Users/fengxuedong/Desktop/MTS_feature_regression/'
    input_filename = file_path+'/data/'+'all-bubble_list_2007-2024_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_add_extend_dot_num_with_SW_parameter.csv'
    output_filename = file_path+'/data/'+'all-bubble_list_2007-2024_exact-time-period_and_add-extend_Vx_prep_gt50_0_and_V_prep_gt50_0_add_extend_dot_num_with_SW_parameter_Plasma_Sheet_Backgroud_del_483.csv'

    bubble_list_with_SW = pd.read_csv(input_filename)
    print(f"bubble_list_with_SW columns:{bubble_list_with_SW.columns}")
    
    bubble_list_with_SW = bubble_list_with_SW[bubble_list_with_SW['Unnamed: 0']!= 483]
    bubble_list_with_SW = bubble_list_with_SW.reset_index(drop=True)
    print(f"len(bubble_list_with_SW):{len(bubble_list_with_SW)}")
    
    # for column in ['Bz_mean','Np_mean','Vp_mean']:
    #     mean_value = bubble_list_with_SW[column].mean()
        # bubble_list_with_SW[column].fillna(mean_value, inplace=True)
    bubble_list_with_SW['Pp_background'] = None
    bubble_list_with_SW['Tp_background'] = None
    bubble_list_with_SW['Np_background'] = None

    for i in range(0,len(bubble_list_with_SW)):
        xgsm = bubble_list_with_SW['Pos_X'][i]
        ygsm = bubble_list_with_SW['Pos_Y'][i]
        n_sw = bubble_list_with_SW['Np_mean'][i]
        v_sw = bubble_list_with_SW['Vp_mean'][i]
        imf_bz = bubble_list_with_SW['Bz_mean'][i]
        # print(f"i:{i},xgsm:{xgsm},ygsm{ygsm}")
        bubble_list_with_SW['Pp_background'][i] = Pps_tm2003_new(xgsm, ygsm, n_sw, v_sw, imf_bz)
        bubble_list_with_SW['Tp_background'][i] = Tps_tm2003_new(xgsm, ygsm, n_sw, v_sw, imf_bz)
        bubble_list_with_SW['Np_background'][i] = Nps_tm2003_new(xgsm, ygsm, n_sw, v_sw, imf_bz)

    bubble_list_with_SW.to_csv(output_filename)
