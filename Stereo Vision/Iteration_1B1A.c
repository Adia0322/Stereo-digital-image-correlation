/* Iteration 2022/3/17 13:53*/
/* Calculate Gvalue_g and Correlation_sum */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* Mathematical formula */
/* Parameters */
#define Size 31 
#define Scan 31
#define Len (int)((Size-1)/2)
#define Length (int)(0.5*(Size-1)+0.5*(Scan-1))
/* Functions */
double Bicubic(double x, double y, double A_re[][4]);

/* =============== Construct Correlation[] ===================*/
void Gvalue_g(double Gvalue_g[][Size], double Cubic_coef[2*Length+1][2*Length+1][16],\
 double Warp[3][3])
{
	int i, j, u, v, a1, a2;
	double k, m, x, y;
	double A_re[4][4];
	
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			/* Distance between reference point and target point (local coordinate)*/
			k = Warp[0][0]*(i-Len) + Warp[0][1]*(j-Len) + Warp[0][2];
			m = Warp[1][0]*(i-Len) + Warp[1][1]*(j-Len) + Warp[1][2];
			if (k<0)
			{
				a1 = Length + (int)(k-1);
				x = k - (int)(k-1);
			}
			else
			{
				a1 = Length + (int)(k);
				x = k - (int)(k);
			}
			if (m<0)
			{
				a2 = Length + (int)(m-1);
				y = m - (int)(m-1);
			}
			else
			{
				a2 = Length + (int)(m);
				y = m - (int)(m);
			}
			/* Check if a1, a2 is in area */
			if (a1 > 2*Length)
			{
				a1 = 2*Length;
			}
			if (a1 < 0)
			{
				a1 = 0;
			}
			if (a2 > 2*Length)
			{
				a2 = 2*Length;
			}
			if (a2 < 0)
			{
				a2 = 0;
			}
			
			/* Construct Bicubic coefficient matrix */ 
			int count=0;
			for (u=0;u<4;u++)
			{
				for (v=0;v<4;v++)
				{
					/* columnÀu¥ý¶ñ¥R */ 
					A_re[v][u] = Cubic_coef[a1][a2][count];
					count += 1;
				}
			}
			/* Calculate new target subset using interpolation */
			Gvalue_g[i][j] = Bicubic(x, y, A_re);
		}
	}
}

/* ================ Compute Correlation_sum =================*/
void CorrSum(double Correlation_sum[6], double dF_dP[Size][Size],\
 double J_1B1A[Size][Size][6])
{
	int i, j;
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Correlation_sum[0] += J_1B1A[i][j][0]*dF_dP[i][j];
			Correlation_sum[1] += J_1B1A[i][j][1]*dF_dP[i][j];
			Correlation_sum[2] += J_1B1A[i][j][2]*dF_dP[i][j];
			Correlation_sum[3] += J_1B1A[i][j][3]*dF_dP[i][j];
			Correlation_sum[4] += J_1B1A[i][j][4]*dF_dP[i][j];
			Correlation_sum[5] += J_1B1A[i][j][5]*dF_dP[i][j];
		}
	}
}


/* =============== Functions =============== */
double Bicubic(double x, double y, double A_re[][4])
{
	double GrayValue;
	GrayValue =\
	A_re[0][0] + A_re[0][1]*y + A_re[0][2]*y*y + A_re[0][3]*y*y*y +\
	A_re[1][0]*x + A_re[1][1]*x*y + A_re[1][2]*x*y*y + A_re[1][3]*x*y*y*y +\
	A_re[2][0]*x*x + A_re[2][1]*x*x*y + A_re[2][2]*x*x*y*y + A_re[2][3]*x*x*y*y*y +\
	A_re[3][0]*x*x*x + A_re[3][1]*x*x*x*y + A_re[3][2]*x*x*x*y*y + A_re[3][3]*x*x*x*y*y*y;
	
	return GrayValue;
}

