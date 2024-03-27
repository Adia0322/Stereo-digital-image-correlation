/*  PSO_ICGN_1B2B 2022.03.06 23:22 */ 
/* scan must be odd number, ex:101 */ 
/* Coefficient criterion: ZNCC */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/* 數學式 */
#define square(x) (x)*(x)
#define mean(x) x/(Size*Size)
/* 圖像設定 */
#define img_row 480 
#define img_column 640 
/* DIC 參數設定 */ 
#define Size 31 
#define SizeHalf (Size-1)/2 
#define scan 31        /* 方形掃描區域邊長 */ 
/* PSO 參數設定 */ 
#define Population 100 
#define Dimension 2
#define Iteration 20
#define Iter_reciprocal 1/Iteration 
#define ArraySize_Pini 5      /* 設定n*n方陣點的初始固定位置(n為奇數)，使初始搜尋更加均勻，但注意Population須足夠!! */ 
#define FixedPointRange scan/2   /* 設定n*n方陣點之範圍 (方陣邊長) */ 
#define ArrayInterval (FixedPointRange)/(ArraySize_Pini-1) 
#define Array_Start ArrayInterval*(ArraySize_Pini-1)/2
#define Boundary_Length 0.5*(scan-1)
#define Vmax 0.5*Boundary_Length
#define Vini 0.2*Vmax
#define W_upper 0.9 
#define W_lower 0.4 
#define Decrease_factor 1 
#define Increase_factor 1.05
#define Cognition_factor 1.0 
#define Social_factor 1.0 

/* functions */
double Cost_function(int Pi_u, int Pi_v, int Object_point[], int img_aft[][img_column],\
                     int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[]);
double GRandom(void);

/* construct C matrix (Main)  */ 
void SCAN(int img_aft[][img_column], int img_aft_sub[][Size], int img_bef_sub[][Size],\
          double Mean_bef[], int Object_point[], int Displacement[], double CoefValue[])
{
	int i, j, k, m, x, y, Pi_u_ini, Pi_v_ini, Pi_u, Pi_v, Count_u=0, Count_v=0; /*  */ 
	int max_index = 0; 
	double Pbest[Population][Dimension], Gbest[Dimension];   /* Gbest[0] = x, Gbest[1]  = y */
	double upper_bounds[2]={Boundary_Length, Boundary_Length}, lower_bounds[2]={-Boundary_Length, -Boundary_Length}; 
	double Pi[Population][Dimension], Vi[Population][Dimension];
	double Cost_initial, Cost, max_value_Gbest=-1e+9, max_value_Pbest[Population]; /* 設定max_value_Gbest時數值盡量小 */ 
	double Gbest_u, Gbest_v;
	
	srand(time(NULL));   /*亂數前置設定 */
	
	for (i=0;i<Population;i++) /* i: 點編號 */ 
	{
		/* 設定固定速度與位置之點 數量:(ArraySize_Pini)*(ArraySize_Pini) */ 
		if (i<(ArraySize_Pini*ArraySize_Pini))  /* 確認粒子編號小於欲固定之粒子數量 */ 
		{
			Vi[i][0] = Vini*(GRandom()*2-1); /* 創造 -1~1 之小數  note: Vini*(-1.0 ~ 1.0)  */
			Vi[i][1] = Vini*(GRandom()*2-1);
			if (Count_u == ArraySize_Pini)
			{
				Count_u = 0;
				Count_v += 1;
			}
			Pi[i][0] = -Array_Start + Count_u*ArrayInterval;
			Pi[i][1] = -Array_Start + Count_v*ArrayInterval;
			
			Pbest[i][0] = Pi[i][0]; /* 將第i個點的u、v值(j=0為u, j=1為v)紀錄，在初始化Pi[i][j]即為個人經驗最佳位置Pbest[i][j] */
			Pbest[i][1] = Pi[i][1];
			
			/* Calculate SSD (sum of squared differences) */
			Pi_u_ini = (int)Pi[i][0]; /* 強制轉型只去掉小數，不會四捨五入 */
			Pi_v_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_u_ini, Pi_v_ini, Object_point, img_aft,\
			                             img_aft_sub, img_bef_sub, Mean_bef);
			
			/* Individual best value */
			max_value_Pbest[i] = Cost_initial; /* 一開始個人最佳經驗只有1個 */
			
			/* Global best value */
			if (Cost_initial>max_value_Gbest)
			{
				max_value_Gbest = Cost_initial;
				max_index = i;
				
				Gbest[0] = Pi[max_index][0];
				Gbest[1] = Pi[max_index][1];
			}
			
			Count_u += 1;
		}
		
		else
		{
			/* 設定隨機速度與位置之點 */
			for (j=0;j<Dimension;j++)
			{
				Vi[i][j] = Vini*(GRandom()*2-1); /* Vini*(-1.0 ~ 1.0) */
				Pi[i][j] = lower_bounds[j] + 0.5*Boundary_Length +\
				           0.5*GRandom()*(upper_bounds[j]-lower_bounds[j]); /* 注意!! Vi、Pi為浮點數型態 */
				
				Pbest[i][j] = Pi[i][j]; /* 將第i個點的u、v值(j=0為u, j=1為v)紀錄，在初始化Pi[i][j]即為個人經驗最佳位置Pbest[i][j] */
			}
			/* Calculate SSD (sum of squared differences) */
			Pi_u_ini = (int)Pi[i][0]; /* 強制轉型只去掉小數，不會四捨五入 */ 
			Pi_v_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_u_ini, Pi_v_ini, Object_point, img_aft,\
			                             img_aft_sub, img_bef_sub, Mean_bef);  /* 注意!! 這裡不受Dimension而變化，若要更換維度請注意!!!*/ 
			
			/* Individual best value */
			max_value_Pbest[i] = Cost_initial; /* 一開始個人最佳經驗只有1個 */ 
			
			/* Global best value */
			if (Cost_initial>max_value_Gbest)
			{
				max_value_Gbest = Cost_initial;  /* max_value_Gbest為群體最佳位置之值，將在之後每次迭代中更新 */
				max_index = i; /* max_index為群體最佳位置之點編號，群體最佳位置: (u,v) = (Pi[max_index,0] , Pi[max_index,1]) */
				
				Gbest[0] = Pi[max_index][0]; /* Gbest[0]表示群體最佳位置x值，Gbest[1]則為y值。   注意!! 這裡不受Dimension而變化，若要更換維度請注意!!!*/
				Gbest[1] = Pi[max_index][1];
			}
		}
	}
	
	/* Start iteration */ 
	
	for (k=0;k<Iteration;k++)
	{
		for (i=0;i<Population;i++)
		{
			for (j=0;j<Dimension;j++)
			{
				Vi[i][j] = (W_upper-(k+1)*(W_upper-W_lower)*Iter_reciprocal)*Vi[i][j] +\
				            Cognition_factor*GRandom()*(Pbest[i][j]-Pi[i][j]) +\
				            Social_factor*GRandom()*(Gbest[j]-Pi[i][j]);
				if (Vi[i][j]>Vmax)
				{
					Vi[i][j] = Vmax; /* 限制速度上限 */ 
				}
				if (Vi[i][j]<-Vmax)
				{
					Vi[i][j] = -Vmax;
				}
				
				Pi[i][j] += Vi[i][j];  
				
				if (Pi[i][j]>upper_bounds[j]) /* restrict boundary */ 
				{
					Pi[i][j] = upper_bounds[j];
				}
				
				if (Pi[i][j]<lower_bounds[j])
				{
					Pi[i][j] = lower_bounds[j];
				}
			}
			
			/* Calculate ZNCC (sum of squared differences) */
			Pi_u = (int)Pi[i][0]; /* array only accept integer as Argument */
			Pi_v = (int)Pi[i][1];
			Cost = Cost_function(Pi_u, Pi_v, Object_point, img_aft,\
			                     img_aft_sub, img_bef_sub, Mean_bef);
			
			/* Individual best value */
			if (Cost>max_value_Pbest[i]) /* 每次迭代Pbest更新為個人經驗最佳位置 */
			{
				max_value_Pbest[i] = Cost;
				Pbest[i][0]=Pi[i][0];
				Pbest[i][1]=Pi[i][1];
				
				/* Global best value */
				if (Cost>max_value_Gbest)
				{
					max_value_Gbest = Cost;
					max_index = i;
					
					Gbest[0] = Pi[max_index][0];
					Gbest[1] = Pi[max_index][1];
				}
			}
			/* sensor */
			/*sensor[i][0] = (int)Pi[i][0];*/
			/*sensor[i][1] = (int)Pi[i][1];*/
		}
		
	}
	/*sensor_coef[0] = Cost_function(0, 0, Object_point, img_aft, img_aft_sub, img_bef_sub);*/
	/* Output Result */
	Gbest_u = Gbest[0];
	Gbest_v = Gbest[1];
	Displacement[0] = (int)Gbest_u; /* vertical (down:+) */
	Displacement[1] = (int)Gbest_v; /* horizontal (right:+) */
	CoefValue[0] = max_index; /* The point index of the global maximum value */
	CoefValue[1] = max_value_Gbest; /* The global maximum value */
}


/*============================ Functions ==============================*/
/* 依據 ZNCC(zero-normalized cross-correlation)相關係數準則 計算相關係數 (-1 ~ +1) */
double Cost_function(int Pi_u, int Pi_v, int Object_point[], int img_aft[][img_column],\
                     int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[])   
{
	int i, j, Aft_sub_sum=0;
	double Mean_aft, Sum_Numerator=0.0, Sum_Denominator_bef=0.0, Sum_Denominator_aft=0.0, coef=0.0;
	/* Construct img_aft_sub */
	for (i=0;i<Size;i++) 
	{
		for (j=0;j<Size;j++)
		{
			img_aft_sub[i][j] =\
			 img_aft[i - SizeHalf + Pi_u + Object_point[0]][j - SizeHalf + Pi_v + Object_point[1]]; /*   */ 
		}
	}
	/* Mean of img_aft_sub */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Aft_sub_sum+=img_aft_sub[i][j];
		}
	}
	Mean_aft=mean(Aft_sub_sum); /* mean function is defined by macro (#define)   */ 
	/* Substract its mean, comopute its sqrt and sum */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Sum_Numerator+=(img_bef_sub[i][j] - Mean_bef[0])*(img_aft_sub[i][j] - Mean_aft);
			Sum_Denominator_bef+=square(img_bef_sub[i][j] - Mean_bef[0]);
			Sum_Denominator_aft+=square(img_aft_sub[i][j] - Mean_aft);
		}
	}
	coef = Sum_Numerator/(sqrt(Sum_Denominator_bef*Sum_Denominator_aft));
	return coef;
}

/* Generate random number */
double GRandom(void)
{
	double i = fmod(rand(),1000.0)/1000.0;
	return i;
}
