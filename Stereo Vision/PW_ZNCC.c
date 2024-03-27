/* 2D-DIC 2023/01/4  PW + ZNCC */
#include <math.h> 
#define Size 31
#define size 15 
#define Scan 31 /* 掃描方形邊長 */ 
#define scan (Scan-1)/2   
#define square(x) (x)*(x)
#define mean(x) x/(Size*Size)
/* 圖像設定 */
#define img_row 480
#define img_cloumn 640

/* functions */
int Sub_MSSS(int img_aft_sub[][Size], int img_bef_sub[][Size]);
double ZNCC(int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[]);

/* construct C matrix (Main)  */ 
void SCAN(int img_aft[][img_cloumn], int img_aft_sub[][Size], int img_bef_sub[][Size],\
          double Mean_bef[], int point[], double Cmatrix[][Scan])
{
	int i,j,k,m;
	for (i=-scan;i<scan+1;i++)
	{
		for (j=-scan;j<scan+1;j++)
		{
			for (k=-size;k<size+1;k++)
			{
				for (m=-size;m<size+1;m++)
				{
					img_aft_sub[k+size][m+size] = img_aft[point[0]+i+k][point[1]+j+m];  /* move[0,1] = [move1,move2] */
				}
			}
			Cmatrix[scan+i][scan+j] = ZNCC(img_aft_sub, img_bef_sub, Mean_bef); 
		}
	}
}

/*============================ Functions ==============================*/
/* ======== SSD ====== */ 
int Sub_MSSS(int img_aft_sub[][Size], int img_bef_sub[][Size])
{
	int i, j, Mean, Aft_sub_sum=0, sum=0;
	/* Mean of img_aft_sub */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Aft_sub_sum+=img_aft_sub[i][j];
		}
	}
	Mean=mean(Aft_sub_sum);
	/* Substract its mean, comopute its square and sum */  
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			sum+=square(img_aft_sub[i][j]-Mean-img_bef_sub[i][j]);
		}
	}
	return sum;
}

/* ========= ZNCC ========= */
double ZNCC(int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[])   
{
	int i, j, Aft_sub_sum=0;
	double Mean_aft, Sum_Numerator=0.0, Sum_Denominator_bef=0.0, Sum_Denominator_aft=0.0, coef=0.0;
	/* Mean of img_aft_sub */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Aft_sub_sum += img_aft_sub[i][j];
		}
	}
	Mean_aft = mean(Aft_sub_sum); /* mean function is defined by macro (#define)   */ 
	/* Substract its mean, comopute its sqrt and sum */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Sum_Numerator += (img_bef_sub[i][j] - Mean_bef[0])*(img_aft_sub[i][j] - Mean_aft);
			Sum_Denominator_bef += square(img_bef_sub[i][j] - Mean_bef[0]);
			Sum_Denominator_aft += square(img_aft_sub[i][j] - Mean_aft);
		}
	}
	coef = Sum_Numerator/(sqrt(Sum_Denominator_bef)*sqrt(Sum_Denominator_aft));
	return coef;
}
