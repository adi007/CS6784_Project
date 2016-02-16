#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <string>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream> 
#define sampleFreq 5

using namespace std;

struct sparse
{
	int pos;
	double val;
};

struct dataV
{
	int len;
	struct sparse * dat;
};

struct ClusterInfo
{
	int * members;
	int maxl;
	int lastVisitedRow;
	double clusterSize;
	dataV delta;
};

struct timelineStruct
{
	int lastTouch;
};

double sqr(double a)
{
	return a*a;
}

double abso(double a)
{
	if (a<0) return -a;
	return a;
}
int abso(int a)
{
	if (a < 0) return -a;
	return a;
}

int getRandNum(int n)
{
	return abso((rand() * 59999 + rand()) % n);
}

double getRandFloat()
{
	double a = getRandNum(10000);
	if (rand() % 2 == 0)
		a = -a;
	return a / 10000;
}

double inner_sparse(double *w, dataV x)
{
	double sum = 0;
	for (int i = 0; i < x.len; i++)
		sum += w[x.dat[i].pos] * x.dat[i].val;
	return sum;
}

void fullGradient(dataV* X_dat, double * Y_dat, double * gradAns, int d, int n, void * modelPara, void * additionalStuff)
{
	memset(gradAns, 0, sizeof(double)*d);

	double lambda = ((double *)additionalStuff)[1];

	double * w = (double *)modelPara;

	for (int i = 0; i < n; i++)
	{
		double tmp = inner_sparse(w, X_dat[i]) - Y_dat[i];
		for (int j = 0; j < X_dat[i].len; j++)
			gradAns[X_dat[i].dat[j].pos] += X_dat[i].dat[j].val * tmp;
	}

	for (int i = 0; i < d; i++)
		gradAns[i] /= n;
}

double updateW(double val, timelineStruct &tl, int curT, double * save, int sp, double proximalThres, double shrink, double normalgrad)
{
	int totR = curT - 1 - tl.lastTouch;
	tl.lastTouch = curT - 1;
	if (totR <= 0)
		return val;

	if (proximalThres <= 1e-10)
	{
		//shrinking..
		double t = shrink;
		double progSum = (1.0 - pow(t, totR)) / (1.0 - t);
		save[sp] += val * t* progSum - normalgrad *t / (1.0 - t) * (totR - t*progSum);
		return val*pow(shrink, totR) - normalgrad*t* progSum;
	}

	if (val >= 0)
	{
		if (normalgrad > 0)
		{
			//n+p>=0
			int stepAppr = (int)(val / (normalgrad + proximalThres));
			if (stepAppr >= totR)
			{
				save[sp] += (val - (normalgrad + proximalThres)*(1.0 + totR) / 2.0)*totR;
				return val - (normalgrad + proximalThres)*totR;
			}
			totR -= stepAppr;
			save[sp] += (val - (normalgrad + proximalThres)*(1.0 + stepAppr) / 2.0)*stepAppr;
			val -= (normalgrad + proximalThres)*stepAppr;

			if (normalgrad <= proximalThres)
				//never escape
				return 0;
			totR--;
			val -= normalgrad;
			val += proximalThres;
			if (val >= 0)
				val = 0;
			save[sp] += (val - (normalgrad - proximalThres)*(0.0 + totR) / 2.0)*(totR + 1);
			return val - (normalgrad - proximalThres)*totR;
		}
		else
			//below: tl.normalgrad<=0
		{
			if (normalgrad + proximalThres <= 1e-8)
			{
				save[sp] += (val - (normalgrad + proximalThres)*(1.0 + totR) / 2.0)*totR;
				//-|n|+p<=0, go more and more positive
				return val - (normalgrad + proximalThres)*totR;
			}
			//below:  -|n|+p>0, pull close to 0
			int stepAppr = (int)(val / (normalgrad + proximalThres));
			if (stepAppr >= totR)
			{
				save[sp] += (val - (normalgrad + proximalThres)*(1.0 + totR) / 2.0)*totR;
				return val - (normalgrad + proximalThres)*totR;
			}
			save[sp] += (val - (normalgrad + proximalThres)*(1.0 + stepAppr) / 2.0)*stepAppr;
			//else,never escape
			return 0;
		}
	}
	else
		//val<0
	{
		if (normalgrad < 0)
		{
			// -|n|-p=n-p<0, which is the progress
			// -val/ |n-p|= -val / (p-n)
			int stepAppr = (int)(-val / (-normalgrad + proximalThres));
			if (stepAppr >= totR)
			{
				save[sp] += (val - (normalgrad - proximalThres)*(1.0 + totR) / 2.0)*totR;
				return val - (normalgrad - proximalThres)*totR;
			}
			totR -= stepAppr;
			save[sp] += (val - (normalgrad - proximalThres)*(1.0 + stepAppr) / 2.0)*stepAppr;
			val -= (normalgrad - proximalThres)*stepAppr;
			if (-normalgrad <= proximalThres)
				//never escape
				return 0;
			//below n+p<0
			totR--;
			val -= normalgrad;
			val -= proximalThres;
			if (val <= 0)
				val = 0;
			save[sp] += (val - (normalgrad + proximalThres)*(0.0 + totR) / 2.0)*(totR + 1);
			return val - (normalgrad + proximalThres)*totR;
		}
		else
			//below: tl.normalgrad>=0
		{
			if (normalgrad - proximalThres >= -1e-8)
			{
				save[sp] += (val - (normalgrad - proximalThres)*(1.0 + totR) / 2.0)*totR;
				// n - p >=0, go more and more negative
				return val - (normalgrad - proximalThres)*totR;
			}
			//below:  n-p<0, pull close to 0
			int stepAppr = (int)(-val / (-normalgrad + proximalThres));
			if (stepAppr >= totR)
			{
				save[sp] += (val - (normalgrad - proximalThres)*(1.0 + totR) / 2.0)*totR;
				return val - (normalgrad - proximalThres)*totR;
			}
			save[sp] += (val - (normalgrad - proximalThres)*(1.0 + stepAppr) / 2.0)*stepAppr;
			//else,never escape
			return 0;
		}
	}
}

double linearRegressionObjCal(dataV* X_dat, double * Y_dat, int d, int n, void * modelPara, void * additionalStuff)
{
	double * w = (double *)modelPara;
	double sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += 0.5 / n* sqr(inner_sparse(w, X_dat[i]) - Y_dat[i]);
	}

	double sigma = ((double *)additionalStuff)[0];
	double lambda = ((double*)additionalStuff)[1];

	for (int i = 0; i < d; i++)
		sum += sigma*abso(w[i]) + lambda / 2 * sqr(w[i]);
	return sum;
}

void printToFile(dataV* X_dat, double * Y_dat, int d, int n, double * w, void* additionalStuff, FILE* fout, double totGrad, double &lastPrint)
{
	double curObj = linearRegressionObjCal(X_dat, Y_dat, d, n, w, additionalStuff);
	lastPrint = totGrad;
	fprintf(fout, "%.4lf %.15lf\n", totGrad / n, curObj);
}

void uniVR(dataV* X_dat, double * Y_dat, int d, int n, void * modelPara, void * additionalStuff, ClusterInfo* cBlock, int sCluster, int * belongTo, char * fname)
{
	double sigma = ((double*)additionalStuff)[0];
	double lambda = ((double *)additionalStuff)[1];
	double eta = ((double *)additionalStuff)[2];
	double beta = ((double *)additionalStuff)[3];
	double stopErr = ((double *)additionalStuff)[4];
	double lastVal = 10000000000;
	double* wtilde = new double[d];
	double* utilde = new double[d];
	double * save = new double[d];
	double * w = (double *)modelPara;
	int *metBefore = new int[d];
	double thres = abso(eta*sigma);
	int m = n / 4 + 1;
	memcpy(wtilde, w, sizeof(double)*d);

	timelineStruct * timeline = new timelineStruct[d];
	memset(timeline, 0, sizeof(timelineStruct)*d);

	FILE * fout;
	fopen_s(&fout, fname, "w");
	double totGrad = 0;
	double lastPrint = 0;

	while (true)
	{
		memset(save, 0, sizeof(double)*d);

		//below, compute the actual full gradient, utilde is not sparse.
		fullGradient(X_dat, Y_dat, utilde, d, n, (void*)wtilde, additionalStuff);
		totGrad += n;
		for (int i = 0; i < sCluster; i++)
		{
			memset(cBlock[i].delta.dat, 0, sizeof(sparse)*cBlock[i].maxl);
			cBlock[i].lastVisitedRow = -1;
		}
		memset(metBefore, 0, sizeof(int)*d);
		int counter = 0;


		for (int i = 0; i < d; i++)
			timeline[i].lastTouch = -1;


		for (int t = 0; t < m; t++)
		{
			counter++;
			int randomRow;
			randomRow = getRandNum(n);
			int i = belongTo[randomRow];

			// all coordinates related to j are updated. 
			for (int j = 0; j < X_dat[randomRow].len; j++)
				w[X_dat[randomRow].dat[j].pos] = updateW(w[X_dat[randomRow].dat[j].pos], timeline[X_dat[randomRow].dat[j].pos], t, save, X_dat[randomRow].dat[j].pos,
					eta*sigma, 1.0 - eta*lambda, eta*utilde[X_dat[randomRow].dat[j].pos]);

			// all coordinates realted to j_i are also updated. 
			int vR = cBlock[i].lastVisitedRow;
			if (vR >= 0)
			{
				for (int j = 0; j < X_dat[vR].len; j++)
				{
					w[X_dat[vR].dat[j].pos] = updateW(w[X_dat[vR].dat[j].pos], timeline[X_dat[vR].dat[j].pos], t, save, X_dat[vR].dat[j].pos,
						eta*sigma, 1.0 - eta*lambda, eta*utilde[X_dat[vR].dat[j].pos]);
				}
			}

			//compute g1 and g2 here. 
			double t1 = inner_sparse(w, X_dat[randomRow]) - Y_dat[randomRow];
			double t2 = inner_sparse(wtilde, X_dat[randomRow]) - Y_dat[randomRow];

			// - eta(utilde + \nabla_ f -\nabla_i f')
			for (int j = 0; j < X_dat[randomRow].len; j++)
			{
				sparse curS = X_dat[randomRow].dat[j];
				double g;
				g = curS.val*t1;
				g -= curS.val*t2;
				// w - eta( utilde + g1-g2)
				w[curS.pos] -= eta*(curS.val*(t1 - t2) + utilde[curS.pos]);
				metBefore[curS.pos] = counter;
			}
			// - eta(- delta) only if vR>=0
			// also remove the corresponding part in utilde
			if (vR >= 0)
			{
				for (int j = 0; j < X_dat[vR].len; j++)
				{
					sparse curS = X_dat[vR].dat[j];
					// w + eta * old delta
					if (metBefore[curS.pos] == counter)
						w[curS.pos] -= eta*(-cBlock[i].delta.dat[j].val);
					else
						w[curS.pos] -= eta*(utilde[curS.pos] - cBlock[i].delta.dat[j].val);
					// utilde + (old delta)* size/n
					utilde[curS.pos] -= (cBlock[i].delta.dat[j].val) *cBlock[i].clusterSize / n;
				}
			}

			if (vR >= 0)
			{
				for (int j = 0; j < X_dat[vR].len; j++)
				{
					sparse curS = X_dat[vR].dat[j];
					if (sigma>1e-10)
					{
						if (w[curS.pos] > thres)
							w[curS.pos] -= thres;
						else if (w[curS.pos] < -thres)
							w[curS.pos] += thres;
						else w[curS.pos] = 0;
					}
					else
						w[curS.pos] *= 1.0 - eta*lambda;
					timeline[curS.pos].lastTouch = t;
					save[curS.pos] += w[curS.pos];
				}
			}
			for (int j = 0; j < X_dat[randomRow].len; j++)
			{
				sparse curS = X_dat[randomRow].dat[j];
				// if this coorindate is visited before, skip this one. 
				if (timeline[curS.pos].lastTouch == t)
					continue;
				if (sigma>1e-10)
				{
					if (w[curS.pos] > thres)
						w[curS.pos] -= thres;
					else if (w[curS.pos] < -thres)
						w[curS.pos] += thres;
					else w[curS.pos] = 0;
				}
				else
					w[curS.pos] *= 1.0 - eta*lambda;
				timeline[curS.pos].lastTouch = t;
				save[curS.pos] += w[curS.pos];
			}

			//update the lastVisitedRow
			cBlock[i].lastVisitedRow = randomRow;
			//update delta as g1-g2
			// update utilde as well. 
			for (int j = 0; j < X_dat[randomRow].len; j++)
			{
				sparse curS = X_dat[randomRow].dat[j];
				//update delta here.  = g1-g2
				cBlock[i].delta.dat[j].val = curS.val*(t1 - t2);
				// utilde + (delta)* size/n
				utilde[curS.pos] += (cBlock[i].delta.dat[j].val) *cBlock[i].clusterSize / n;
			}

			totGrad++;
			if (totGrad > lastPrint + n / sampleFreq)
			{
				for (int j = 0; j < d; j++)
					w[j] = updateW(w[j], timeline[j], t + 1, save, j, eta*sigma, 1.0 - eta*lambda, eta*utilde[j]);
				printToFile(X_dat, Y_dat, d, n, w, additionalStuff, fout, totGrad, lastPrint);
			}

			/*
			for (int j = 0; j < d;j++)
			{
			w[j] = updateW(w[j], timeline[j], t+1, save, j, eta*sigma, 1.0-eta*lambda, eta*utilde[j]);
			//timeline[j].lastTouch = t;
			}*/

		}
		for (int j = 0; j < d; j++)
		{
			w[j] = updateW(w[j], timeline[j], m, save, j, eta*sigma, 1.0 - eta*lambda, eta*utilde[j]);
			wtilde[j] = save[j] / m;
		}

		m = int(m* beta);

		double curObj = linearRegressionObjCal(X_dat, Y_dat, d, n, modelPara, additionalStuff);

		printf("%.10lf  eta=%.5lf  w0=%.5lf  w1=%.5lf  w2=%.5lf\n", curObj, eta, w[0], w[1], w[2]);


		if ((abso(curObj - lastVal) < stopErr) || (totGrad / 300 >= n))
			break;
		lastVal = curObj;
	}


	delete[] wtilde;
	delete[] utilde;
	delete[] save;
}





int main(int argc, char ** argv)
{
	srand(0);
	FILE * fdata;
	if (argc < 9)
	{
		fprintf(stdout, "main.exe input helper clustering output sigma lambda eta stopperr\n");
		fprintf(stdout, "   0       1    2       3          4      5      6    7     8 \n");
		if (argc < 5)
			return 0;
		fprintf(stdout, "Now using default.\n");
	}

	fopen_s(&fdata, argv[1], "r");


	FILE* fhelp;
	fopen_s(&fhelp, argv[2], "r");
	int n;
	char label[100];
	fscanf_s(fhelp, "%i\n", &n);
	fscanf_s(fhelp, "%s", &label, 100);
	dataV * dataM = new dataV[n];
	double * yVec = new double[n];


	int d = 0;
	double avgLen = 0;
	for (int iter = 0; iter < n; iter++)
	{
		int y;
		char tmpLabel[100];
		if (iter > 0) fscanf_s(fdata, "\n");
		fscanf_s(fdata, "%s", &tmpLabel, 100);

		if (strcmp(tmpLabel, label) == 0)
			y = 1;
		else
			y = -1;

		fscanf_s(fhelp, "%i", &(dataM[iter].len));

		dataM[iter].dat = new sparse[dataM[iter].len];
		yVec[iter] = y;
		double sqrlen = 0;
		for (int i = 0; i < dataM[iter].len; i++)
		{
			int position;
			double actualValue;
			fscanf_s(fdata, "%i:%lf", &position, &actualValue);
			if (position >= d)
				d = position + 1;
			dataM[iter].dat[i].pos = position;
			dataM[iter].dat[i].val = actualValue;
			sqrlen += actualValue*actualValue;
		}
		avgLen += sqrt(sqrlen);
	}
	avgLen /= n;
	for (int iter = 0; iter < n; iter++)
		for (int i = 0; i < dataM[iter].len; i++)
			dataM[iter].dat[i].val /= avgLen;
	printf("%.10lf\nDone scaling..!\n", avgLen);
	fclose(fdata);
	fclose(fhelp);

	FILE * fClustering;
	fopen_s(&fClustering, argv[3], "r");
	ClusterInfo * cBlock = new ClusterInfo[n];
	memset(cBlock, 0, sizeof(ClusterInfo)*n);
	int sCluster = 0;
	int* belongTo = new  int[n];
	for (int i = 0; i < n; i++)
	{
		fscanf_s(fClustering, "%i", &belongTo[i]);
		cBlock[belongTo[i]].clusterSize++;
		if (sCluster <= belongTo[i])
			sCluster = belongTo[i] + 1;
	}
	fclose(fClustering);


	double * additionalStuff = new double[10];
	additionalStuff[0] = 0;  //sigma for lasso
	additionalStuff[1] = 0.01; //lambda for ridge
	additionalStuff[2] = 0.3; //eta
	additionalStuff[3] = 2.0; //beta
	additionalStuff[4] = 0.0000000000001; //stop Err

	if (argc > 5)
	{
		sscanf_s(argv[5], "%lf", &additionalStuff[0]);
		sscanf_s(argv[6], "%lf", &additionalStuff[1]);
		sscanf_s(argv[7], "%lf", &additionalStuff[2]);
		sscanf_s(argv[8], "%lf", &additionalStuff[4]);
	}


	//additionally set
	additionalStuff[4] = 1e-12; //stop Err

	double * linearRegressionW = new double[d];
	memset(linearRegressionW, 0, sizeof(double)*d);

	for (int i = 0; i < sCluster; i++)
	{
		cBlock[i].members = new int[(int)(cBlock[i].clusterSize + 0.01)];
		cBlock[i].clusterSize = 0;
		cBlock[i].delta.len = 0;
	}
	for (int i = 0; i < n; i++)
	{
		int j = belongTo[i];
		cBlock[j].members[(int)(cBlock[j].clusterSize + 0.01)] = i;
		cBlock[j].maxl = max(cBlock[j].maxl, dataM[i].len);
		cBlock[j].clusterSize++;
	}
	for (int i = 0; i < sCluster; i++)
	{
		cBlock[i].delta.dat = 0;
		cBlock[i].delta.dat = new sparse[cBlock[i].maxl];
	}

	uniVR(dataM, yVec, d, n, linearRegressionW, additionalStuff, cBlock, sCluster, belongTo, argv[4]);

	return 0;
}
