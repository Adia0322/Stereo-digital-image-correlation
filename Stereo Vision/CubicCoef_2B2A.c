/* Interpolation_Cubic_Function 2022.3.19 16:37 */
/* 2B2A */
void CubicCoef(double Cubic_Xinv[16][16], int Length[], int Img[][2*Length[0]+1+3],\
double Coef[][2*Length[0]+1][16])
 {
 	int i, j, k, m, u, v, count;
 	int Length2 = 2*Length[0]+1;
 	int Gvalue[16];
 	for (i=0;i<(Length2);i++){
 		for (j=0;j<(Length2);j++){
 			count = 0;
			 for (k=0;k<4;k++){
 				for (m=0;m<4;m++){
 					Gvalue[count] = Img[i+m][j+k];
 					count += 1;
				 }
			 }
			 for (u=0;u<16;u++){
			 	for (v=0;v<16;v++){
			 		Coef[i][j][u] += Cubic_Xinv[u][v]*Gvalue[v];
				 }
			 }
		 }
	 }
 }
