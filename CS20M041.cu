#include <stdio.h>
#include <iostream>
#include <string>
#include <cuda.h>
#include <thrust/sort.h>

using namespace std;

__global__ void fill(int *p, int *train, int size){

	int id = blockIdx.x * 1024 + threadIdx.x;

	if(id<size){
		for(int i=0; i<25; i++){
			for(int j=0; j<50; j++){
				p[id*25*50 + i*50 + j] = train[id*28+3+i];
			}
		}
	}
}

__global__ void fill1(int reqst, int *p, int *q){

	int id = blockIdx.x*1024+threadIdx.x;
	if(id<reqst){
		p[id]=0;
		q[id]=0;
	}
}

__global__ void book_tickets(int Request, int *seats, int *request, int *sorted, int *train_No, int *coach, int *src, int *dest, int *tickets, int *train_coach, int *train_info, int *result, int *booked, int *free)
{
	int id = blockIdx.x * 1024 + threadIdx.x;
	if(id<Request)
	{

		int reqst, Source, Destination, temp, m, dist, train, train_src;
		//booked[id]=0;
		//result[id]=0;
		reqst = sorted[id];
		if(src[reqst]>dest[reqst]){
			temp = src[reqst];
			src[reqst] = dest[reqst];
			dest[reqst] = temp;
		}
		Source = src[reqst];
		m = coach[reqst];
		train = train_No[reqst];
		Destination = dest[reqst];
		train_src = train_info[train*28];

		if(id==0)
		{
			for(int i=Source; i<Destination; i++){
				if(free[train*25*50 + m*50 +(i-train_src)]<tickets[reqst]){
					booked[id] = 1;
					break;
				}
			}

			if(!booked[id]){
				for(int i = Source; i<Destination; i++){
					free[train*25*50 + m*50 + (i-train_src)]-=tickets[reqst];
				}
				result[reqst] = 1;
				booked[id] = 1;
			}
			booked[id]=1;
		}
		else
		{
			do
			{
				if((train_coach[id]!=train_coach[id-1])||(booked[id-1]))
				{
					for(int i=Source; i<Destination; i++){
						if(free[train*25*50 + m*50 +(i-train_src)]<tickets[reqst]){
							booked[id] = 1;
							break;
						}
					}

					if(!booked[id]){
						for(int i = Source; i<Destination; i++){
							free[train*25*50 + m*50 + (i-train_src)]-=tickets[reqst];
						}
						result[reqst] = 1;
						booked[id] = 1;
					}

					booked[id] = 1;
				}
			}while(!booked[id]);
		}

		if(result[reqst]){
			dist = tickets[reqst]*(Destination - Source);
			atomicAdd(seats,dist);
		}
	}
}

void writeResult(int *result, int request){

	int success_count=0, failure_count = 0;
	for (int i = 0; i < request; ++i)
	{
		if(result[i]){
			cout<<"success"<<"\n";
			success_count+=1;
		}
		else{
			cout<<"failure"<<"\n";
			failure_count+=1;
		}
	}
	cout<<success_count<<" "<<failure_count<<"\n";
}

int main(){

	int N, M, src, dest, train, temp;
	int *train_info, *Dtrain_info, *DTemp;
	int *Result, *DBooked, *DResult;
	int batch, request, *seats;
	int *Request, *Train, *Coach, *Source, *Dest, *Tickets, *Train_coach;
	int *DRequest, *DTrain, *DCoach, *DSource, *DDest, *DTickets, *DTrain_coach;
	
	cin>>N;
	train_info = (int *)malloc(N*28*sizeof(int));
	for (int i = 0; i < N; ++i)
	{
		cin>>train>>M>>src>>dest;
		if(src>dest){
			temp = src;
			src = dest;
			dest = temp;
		}
		train_info[i*28] = src;
		train_info[i*28+1] = dest;
		train_info[i*28+2] = M;
		for(int j=0; j<M; j++){
			cin>>temp>>train_info[i*28+3+j];
		}
		for(int k=M; k<25; k++){
			train_info[i*28+3+k] = 0;
		}
	}

	cudaMalloc(&Dtrain_info, N*28*sizeof(int));
	cudaMemcpy(Dtrain_info, train_info, N*28*sizeof(int), cudaMemcpyHostToDevice);
	int *Free = (int *)malloc(N*25*50*sizeof(int)), *Dfree;
	cudaMalloc(&Dfree, N*25*50*sizeof(int));
	fill<<<(N+1023)/1024, 1024>>>(Dfree, Dtrain_info, N);
	
	cin>>batch;
	for (int i = 0; i < batch; ++i)
	{
		cin>>request;
		Request = (int *)malloc(request*sizeof(int));
		Train = (int *)malloc(request*sizeof(int));
		Coach = (int *)malloc(request*sizeof(int));
		Source = (int *)malloc(request*sizeof(int));
		Dest = (int *)malloc(request*sizeof(int));
		Tickets = (int *)malloc(request*sizeof(int));
		Train_coach = (int *)malloc(request*sizeof(int));
		for(int j=0; j<request; j++){

			cin>>Request[j]>>Train[j]>>Coach[j]>>Source[j]>>Dest[j]>>Tickets[j];
			string str = "0", str1 = to_string(Train[j]), str2 = to_string(Coach[j]),s;

			if((Coach[j]/10)==0){
				s=str+str2;
			}
			else{
				s=str2;
			}
			Train_coach[j] = stoi(str1+s);
		}

		cudaMalloc(&DRequest, request*sizeof(int));
		cudaMalloc(&DTrain, request*sizeof(int));
		cudaMalloc(&DCoach, request*sizeof(int));
		cudaMalloc(&DSource, request*sizeof(int));
		cudaMalloc(&DDest, request*sizeof(int));
		cudaMalloc(&DTickets, request*sizeof(int));
		cudaMalloc(&DTrain_coach, request*sizeof(int));
		cudaMalloc(&DTemp, request*sizeof(int));
		cudaMalloc(&DBooked, request*sizeof(int));
		cudaMalloc(&DResult, request*sizeof(int));
		cudaHostAlloc(&seats, sizeof(int), 0);

		cudaMemcpy(DRequest, Request, request*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(DTemp, Request, request*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(DTrain, Train, request*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(DCoach, Coach, request*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(DSource, Source, request*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(DDest, Dest, request*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(DTickets, Tickets, request*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(DTrain_coach, Train_coach, request*sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		
		thrust::sort_by_key(thrust::device, DTrain_coach, DTrain_coach + request, DTemp);
		cudaDeviceSynchronize();
		fill1<<<(request+1023)/1024, 1024>>>(request, DBooked, DResult);
		book_tickets<<<(request+1023)/1024, 1024>>>(request, seats, DRequest, DTemp, DTrain, DCoach, DSource, DDest, DTickets, DTrain_coach, Dtrain_info, DResult, DBooked, Dfree);
		free(Request);
		free(Train);
		free(Coach);
		free(Source);
		free(Dest);
		free(Tickets);
		free(Train_coach);
		Result = (int *)malloc(request * sizeof(int));
		cudaDeviceSynchronize();
		cudaMemcpy(Result, DResult, request*sizeof(int), cudaMemcpyDeviceToHost);
		
		cudaFree(DRequest);
		cudaFree(DTrain);
		cudaFree(DCoach);
		cudaFree(DSource);
		cudaFree(DDest);
		cudaFree(DTickets);
		cudaFree(DTrain_coach);
		cudaFree(DTemp);
		cudaFree(DBooked);
		cudaFree(DResult);
		writeResult(Result, request);
		cout<<*seats<<"\n";
		free(Result);
	}

	cudaFree(Dtrain_info);
	cudaFree(Dfree);
	free(train_info);
	free(Free);
	return 0;
}