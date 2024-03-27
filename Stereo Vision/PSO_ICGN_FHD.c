/*  PSO_ICGN 2022.03.05 21:05 rearranged version*/ 
/* scan must be odd number, ex:101 */ 
/* REARRNAGE */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/* 數學式 */
#define sqrt(x) (x)*(x)
#define mean(x) x/(Size*Size)
/* 圖像設定 */
#define img_row 1080
#define img_cloumn 1920
/* DIC 參數設定 */ 
#define Size 23 
#define SizeHalf (Size-1)/2 
#define scan 61        /* 方形掃描區域邊長 */ 
/* PSO 參數設定 */ 
#define Population 45 
#define Dimension 2
#define Iteration 4 
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
int Cost_function(int Pi_u, int Pi_v, int Object_point[], int img_aft[][img_cloumn],\
                  int img_aft_sub[][Size], int img_bef_sub[][Size]);
float GRandom(void);

/* construct C matrix (Main)  */ 
void SCAN(int img_aft[][img_cloumn], int img_aft_sub[][Size], int img_bef_sub[][Size],\
 int Object_point[], int Displacement[])
{
	int i, j, k, m, x, y, Pi_u_ini, Pi_v_ini, Pi_u, Pi_v, Count_u=0, Count_v=0; /*  */ 
	int min_index = 0; 
	float Pbest[Population][Dimension], Gbest[Dimension];   /* Gbest[0] = x, Gbest[1]  = y */
	float upper_bounds[2]={Boundary_Length, Boundary_Length}, lower_bounds[2]={-Boundary_Length, -Boundary_Length}; 
	float Pi[Population][Dimension], Vi[Population][Dimension];
	float Cost_initial, Cost, min_value_Gbest=9e+10, min_value_Pbest[Population]; /* 設定min_value_Gbest時數值盡量大 */ 
	float Gbest_u, Gbest_v;
	
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
			Cost_initial = Cost_function(Pi_u_ini, Pi_v_ini, Object_point,\
			                             img_aft, img_aft_sub, img_bef_sub);
			
			/* Individual best value */
			min_value_Pbest[i] = Cost_initial; /* 一開始個人最佳經驗只有1個 */
			
			/* Global best value */
			if (Cost_initial<min_value_Gbest)
			{
				min_value_Gbest = Cost_initial;
				min_index = i;
				
				Gbest[0] = Pi[min_index][0];
				Gbest[1] = Pi[min_index][1];
			}
			
			Count_u += 1;
		}
		
		else
		{
			/* 設定隨機速度與位置之點 */
			for (j=0;j<Dimension;j++)
			{
				Vi[i][j] = Vini*(GRandom()*2-1); /* Vini*(-1.0 ~ 1.0) */
				Pi[i][j] = lower_bounds[j] + 0.5*Boundary_Length + 0.5*GRandom()*(upper_bounds[j]-lower_bounds[j]); /* 注意!! Vi、Pi為浮點數型態 */
				
				Pbest[i][j] = Pi[i][j]; /* 將第i個點的u、v值(j=0為u, j=1為v)紀錄，在初始化Pi[i][j]即為個人經驗最佳位置Pbest[i][j] */
			}
			/* Calculate SSD (sum of squared differences) */
			Pi_u_ini = (int)Pi[i][0]; /* 強制轉型只去掉小數，不會四捨五入 */ 
			Pi_v_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_u_ini, Pi_v_ini, Object_point,\
			                             img_aft, img_aft_sub, img_bef_sub);  /* 注意!! 這裡不受Dimension而變化，若要更換維度請注意!!!*/ 
			
			/* Individual best value */
			min_value_Pbest[i] = Cost_initial; /* 一開始個人最佳經驗只有1個 */ 
			
			/* Global best value */
			if (Cost_initial<min_value_Gbest)
			{
				min_value_Gbest = Cost_initial;  /* min_value_Gbest為群體最佳位置之值，將在之後每次迭代中更新 */
				min_index = i; /* min_index為群體最佳位置之點編號，群體最佳位置: (u,v) = (Pi[min_index,0] , Pi[min_index,1]) */
				
				Gbest[0] = Pi[min_index][0]; /* Gbest[0]表示群體最佳位置x值，Gbest[1]則為y值。   注意!! 這裡不受Dimension而變化，若要更換維度請注意!!!*/
				Gbest[1] = Pi[min_index][1];
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
				Vi[i][j] = (W_upper-(k+1)*(W_upper-W_lower)*Iter_reciprocal)*Vi[i][j]+\
				            Cognition_factor*GRandom()*(Pbest[i][j]-Pi[i][j])+\
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
			
			/* Calculate SSD (sum of squared differences) */
			Pi_u = (int)Pi[i][0]; /* array only accept integer as Argument */
			Pi_v = (int)Pi[i][1];
			Cost = Cost_function(Pi_u, Pi_v, Object_point, img_aft, img_aft_sub, img_bef_sub);
			
			/* Individual best value */
			if (Cost<min_value_Pbest[i]) /* 每次迭代Pbest更新為個人經驗最佳位置 */
			{
				min_value_Pbest[i] = Cost;
				Pbest[i][0]=Pi[i][0];
				Pbest[i][1]=Pi[i][1];
				
				/* Global best value */
				if (Cost<min_value_Gbest)
				{
					min_value_Gbest = Cost;
					min_index = i;
					
					Gbest[0] = Pi[min_index][0];
					Gbest[1] = Pi[min_index][1];
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
	Displacement[2] = (int)min_index; /* The point index of the global minimum value */
	Displacement[3] = (int)min_value_Gbest; /* The global minimum value */
}

/*============================ Functions ==============================*/
/* 依據SSD(sum of squared differences)相關係數準則 計算相關係數 */
int Cost_function(int Pi_u, int Pi_v, int Object_point[], int img_aft[][img_cloumn],\
                  int img_aft_sub[][Size], int img_bef_sub[][Size])   
{
	int sum=0, Mean, Aft_sub_sum=0;
	int i, j, k, m, TEMP;
	/* Construct img_aft_sub */
	for (i=0;i<Size;i++) /* i似乎無法是負值? */ 
	{
		for (j=0;j<Size;j++)
		{
			TEMP = img_aft[i - SizeHalf + Pi_u + Object_point[0]][j - SizeHalf + Pi_v + Object_point[1]]; /*   */ 
			img_aft_sub[i][j] = TEMP;
		}
	}
	/* Mean of img_aft_sub */
	for (k=0;k<Size;k++)
	{
		for (m=0;m<Size;m++)
		{
			Aft_sub_sum+=img_aft_sub[k][m];
		}
	}
	Mean=mean(Aft_sub_sum); /* mean function is defined by macro (#define)    注意!!! mean() 是float被int除 可能有失敗風險??   */ 
	/* Substract its mean, comopute its sqrt and sum */
	for (k=0;k<Size;k++)
	{
		for (m=0;m<Size;m++)
		{
			sum += sqrt(img_aft_sub[k][m]-Mean-img_bef_sub[k][m]);
		}
	}
	return sum;
}

/* Generate random number */
float GRandom(void)
{
	float i = fmod(rand(),1000.0)/1000.0;
	return i;
}
