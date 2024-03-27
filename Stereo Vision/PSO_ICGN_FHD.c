/*  PSO_ICGN 2022.03.05 21:05 rearranged version*/ 
/* scan must be odd number, ex:101 */ 
/* REARRNAGE */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/* �ƾǦ� */
#define sqrt(x) (x)*(x)
#define mean(x) x/(Size*Size)
/* �Ϲ��]�w */
#define img_row 1080
#define img_cloumn 1920
/* DIC �ѼƳ]�w */ 
#define Size 23 
#define SizeHalf (Size-1)/2 
#define scan 61        /* ��α��y�ϰ���� */ 
/* PSO �ѼƳ]�w */ 
#define Population 45 
#define Dimension 2
#define Iteration 4 
#define Iter_reciprocal 1/Iteration 
#define ArraySize_Pini 5      /* �]�wn*n��}�I����l�T�w��m(n���_��)�A�Ϫ�l�j�M��[���áA���`�NPopulation������!! */ 
#define FixedPointRange scan/2   /* �]�wn*n��}�I���d�� (��}���) */ 
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
	float Cost_initial, Cost, min_value_Gbest=9e+10, min_value_Pbest[Population]; /* �]�wmin_value_Gbest�ɼƭȺɶq�j */ 
	float Gbest_u, Gbest_v;
	
	srand(time(NULL));   /*�üƫe�m�]�w */
	
	for (i=0;i<Population;i++) /* i: �I�s�� */ 
	{
		/* �]�w�T�w�t�׻P��m���I �ƶq:(ArraySize_Pini)*(ArraySize_Pini) */ 
		if (i<(ArraySize_Pini*ArraySize_Pini))  /* �T�{�ɤl�s���p����T�w���ɤl�ƶq */ 
		{
			Vi[i][0] = Vini*(GRandom()*2-1); /* �гy -1~1 ���p��  note: Vini*(-1.0 ~ 1.0)  */
			Vi[i][1] = Vini*(GRandom()*2-1);
			if (Count_u == ArraySize_Pini)
			{
				Count_u = 0;
				Count_v += 1;
			}
			Pi[i][0] = -Array_Start + Count_u*ArrayInterval;
			Pi[i][1] = -Array_Start + Count_v*ArrayInterval;
			
			Pbest[i][0] = Pi[i][0]; /* �N��i���I��u�Bv��(j=0��u, j=1��v)�����A�b��l��Pi[i][j]�Y���ӤH�g��̨Φ�mPbest[i][j] */
			Pbest[i][1] = Pi[i][1];
			
			/* Calculate SSD (sum of squared differences) */
			Pi_u_ini = (int)Pi[i][0]; /* �j���૬�u�h���p�ơA���|�|�ˤ��J */
			Pi_v_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_u_ini, Pi_v_ini, Object_point,\
			                             img_aft, img_aft_sub, img_bef_sub);
			
			/* Individual best value */
			min_value_Pbest[i] = Cost_initial; /* �@�}�l�ӤH�̨θg��u��1�� */
			
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
			/* �]�w�H���t�׻P��m���I */
			for (j=0;j<Dimension;j++)
			{
				Vi[i][j] = Vini*(GRandom()*2-1); /* Vini*(-1.0 ~ 1.0) */
				Pi[i][j] = lower_bounds[j] + 0.5*Boundary_Length + 0.5*GRandom()*(upper_bounds[j]-lower_bounds[j]); /* �`�N!! Vi�BPi���B�I�ƫ��A */
				
				Pbest[i][j] = Pi[i][j]; /* �N��i���I��u�Bv��(j=0��u, j=1��v)�����A�b��l��Pi[i][j]�Y���ӤH�g��̨Φ�mPbest[i][j] */
			}
			/* Calculate SSD (sum of squared differences) */
			Pi_u_ini = (int)Pi[i][0]; /* �j���૬�u�h���p�ơA���|�|�ˤ��J */ 
			Pi_v_ini = (int)Pi[i][1];
			Cost_initial = Cost_function(Pi_u_ini, Pi_v_ini, Object_point,\
			                             img_aft, img_aft_sub, img_bef_sub);  /* �`�N!! �o�̤���Dimension���ܤơA�Y�n�󴫺��׽Ъ`�N!!!*/ 
			
			/* Individual best value */
			min_value_Pbest[i] = Cost_initial; /* �@�}�l�ӤH�̨θg��u��1�� */ 
			
			/* Global best value */
			if (Cost_initial<min_value_Gbest)
			{
				min_value_Gbest = Cost_initial;  /* min_value_Gbest���s��̨Φ�m���ȡA�N�b����C�����N����s */
				min_index = i; /* min_index���s��̨Φ�m���I�s���A�s��̨Φ�m: (u,v) = (Pi[min_index,0] , Pi[min_index,1]) */
				
				Gbest[0] = Pi[min_index][0]; /* Gbest[0]��ܸs��̨Φ�mx�ȡAGbest[1]�h��y�ȡC   �`�N!! �o�̤���Dimension���ܤơA�Y�n�󴫺��׽Ъ`�N!!!*/
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
					Vi[i][j] = Vmax; /* ����t�פW�� */ 
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
			if (Cost<min_value_Pbest[i]) /* �C�����NPbest��s���ӤH�g��̨Φ�m */
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
/* �̾�SSD(sum of squared differences)�����Y�Ʒǫh �p������Y�� */
int Cost_function(int Pi_u, int Pi_v, int Object_point[], int img_aft[][img_cloumn],\
                  int img_aft_sub[][Size], int img_bef_sub[][Size])   
{
	int sum=0, Mean, Aft_sub_sum=0;
	int i, j, k, m, TEMP;
	/* Construct img_aft_sub */
	for (i=0;i<Size;i++) /* i���G�L�k�O�t��? */ 
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
	Mean=mean(Aft_sub_sum); /* mean function is defined by macro (#define)    �`�N!!! mean() �Ofloat�Qint�� �i�঳���ѭ��I??   */ 
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
