import numpy as np

"""
LED Number to Angle in Radians Conversion Function

(For specific type of LED Array and set up!)
"""
def get_LED_angle_in_rad(LED_num):
    ###2022/10/25
    ###2022/10/11
    ###2023/11/07
    ###2024/04/02
#                -180/+180 (LED 138)

#        -90 (105)                    +90 (32)

#                    0 (68)

#138 to 14: 17
#14 to 32: 18
#32 to 50: 18
#50 to 68: 18
#68 to 87: 19
#87 to 105: 18
#105 to 121: 16
#121 to 138: 17


    LED_num=LED_num.astype("float")
    LED_num[(LED_num==-1)|(LED_num==149)|(LED_num==150)]=np.nan ### for dark mode


    LED_num[(2<LED_num)&(LED_num<14)]=(np.pi-(np.pi/4/((143-138+1)+(14-2))*(143-138+1)))-((LED_num[(2<LED_num)&(LED_num<14)]-2)*(np.pi/4/((143-138+1)+(14-2))))
    LED_num[(14<=LED_num)&(LED_num<32)]=3*np.pi/4-((LED_num[(14<=LED_num)&(LED_num<32)]-14)*(np.pi/4/(32-14)))
    LED_num[(32<=LED_num)&(LED_num<50)]=np.pi/2-((LED_num[(32<=LED_num)&(LED_num<50)]-32)*(np.pi/4/(50-32)))
    LED_num[(50<=LED_num)&(LED_num<68)]=np.pi/4-((LED_num[(50<=LED_num)&(LED_num<68)]-50)*(np.pi/4/(68-50)))
    LED_num[(68<=LED_num)&(LED_num<87)]=0-((LED_num[(68<=LED_num)&(LED_num<87)]-68)*(np.pi/4/(87-68)))
    LED_num[(87<=LED_num)&(LED_num<105)]=-np.pi/4-((LED_num[(87<=LED_num)&(LED_num<105)]-87)*(np.pi/4/(105-87)))
    LED_num[(105<=LED_num)&(LED_num<121)]=-np.pi/2-((LED_num[(105<=LED_num)&(LED_num<121)]-105)*(np.pi/4/(121-105)))
    LED_num[(121<=LED_num)&(LED_num<138)]=-3*np.pi/4-((LED_num[(121<=LED_num)&(LED_num<138)]-121)*(np.pi/4/(138-121)))
    LED_num[(138<=LED_num)&(LED_num<=143)]=np.pi-((LED_num[(138<=LED_num)&(LED_num<=143)]-138)*(np.pi/4/((143-138)+(14-3))))

    LED_num[LED_num==68]=0
    LED_num[LED_num==105]=-np.pi/2
    LED_num[LED_num==138]=np.pi
    LED_num[LED_num==32]=np.pi/2
    ## 20230920
    LED_num[LED_num==14]=3*np.pi/4 
    LED_num[LED_num==50]=np.pi/4
    LED_num[LED_num==87]=-np.pi/4
    LED_num[LED_num==121]=-3*np.pi/4

    output_LED_arr=LED_num
    return output_LED_arr
