
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <gl/gl.h>
#include <gl/glu.h>
//#define FREEGLUT_STATIC
#include <gl/freeglut.h>

//Defines
#define INF 2e10f
#define rnd( x )  (x * rand()/RAND_MAX)
#define SPHERES 100
#define BOUNCE_COEFFICIENT .7
#define GRAVITY 100
#define INITIAL_X_SPEED 50
#define INITIAL_Y_SPEED 0
#define NUM_CHUNKS 1
#define BLOCK_SIZE 16

//Function Signatures

//Define Structures
struct Sphere {
	float r, g, b;
	float radius;
	float x, y, z;
	float V_x, V_y; // velocity
	__device__ float hit(float ox, float oy, float *n){ // given the address of a pixel and a place to store data...
		float V_x = ox - x;								// find distance
		float V_y = oy - y;
		if (V_x*V_x + V_y*V_y < radius*radius) {	// if V_x^2 + V_y^2 < radius^2 * 1000/z
			float dz = sqrtf(radius*radius - V_x*V_x - V_y*V_y);	// z distance = sqrt(radius^2 - V_x^2 - V_y^2)
			*n = dz / sqrtf(radius * radius); //creates fade effect
			return dz + z;
		}
		return -INF;
	}

};

//Globals
Sphere *s;							// array of sphere objects at s
Sphere *temp_s;						// array of sphere objects at temp_s
int width;							// window size
int height;							// window size
unsigned char* dev_bitmap;
unsigned char* cpu_bitmap;
float elapsedTime;
int iterations;
int mode;
__constant__ Sphere c_s[SPHERES];	// constant array of spheres
int busy_array[SPHERES];
//Declare CUDA Kernels

cudaError_t cudastream();

__global__ void kernel(unsigned char* bitmap, Sphere *s, int width, int height) { // global version

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y *blockDim.x*gridDim.x;
	float ox = (x - width/2);
	float oy = (y - height/2);
	float r = 0, g = 0, b = 0, a = 0;
	float maxz = -INF;
	for (int i = 0; i<SPHERES; i++)	{
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r*fscale;
			g = s[i].g*fscale;
			b = s[i].b*fscale;
		}
	}
	bitmap[offset*4 + 0] = (int)(r*255);
	bitmap[offset*4 + 1] = (int)(g*255);
	bitmap[offset*4 + 2] = (int)(b*255);
	bitmap[offset*4 + 3] = 255;

}
__global__ void kernel_c(unsigned char* bitmap, int width, int height){ // constant memory version

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y *blockDim.x*gridDim.x;
	float ox = (x - width/2);
	float oy = (y - height/2);
	float r = 0, g = 0, b = 0;// a = 0;
	float maxz = -INF;						// max z is how far away to make calculations?
	for (int i = 0; i<SPHERES; i++)	{		// need to see if a ray hits any sphere
		float n;
		float t = c_s[i].hit(ox, oy, &n);	// hit takes ox, oy, and the address for n... and returns the closest point?
		if (t > maxz) {						
			float fscale = n;
			r = c_s[i].r*fscale;
			g = c_s[i].g*fscale;
			b = c_s[i].b*fscale;
		}
	}
	//if(c_s[i].busy)
	{
	bitmap[offset*4 + 0] = (int)(r*255);
	bitmap[offset*4 + 1] = (int)(g*255);
	bitmap[offset*4 + 2] = (int)(b*255);
	bitmap[offset*4 + 3] = 255;
	}
}
__global__ void kernel_cs(unsigned char* bitmap, int width, int height, int y_offset){ // constant mem, streamed version

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y *blockDim.x*gridDim.x;// + starting_point;
	float ox = (x - width/2);
	float oy = (y + y_offset - height/2);
	float r = 0, g = 0, b = 0;// a = 0;
	float maxz = -INF;						// max z is how far away to make calculations?
	for (int i = 0; i<SPHERES; i++)	{		// need to see if a ray hits any sphere
		float n;
		float t = c_s[i].hit(ox, oy, &n);	// hit takes ox, oy, and the address for n... and returns the closest point?
		if (t > maxz) {						
			float fscale = n;
			r = c_s[i].r*fscale;
			g = c_s[i].g*fscale;
			b = c_s[i].b*fscale;
		}
	}
	//if(c_s[i].busy)
	{
	bitmap[offset*4 + 0] = (int)(r*255);
	bitmap[offset*4 + 1] = (int)(g*255);
	bitmap[offset*4 + 2] = (int)(b*255);
	bitmap[offset*4 + 3] = 255;
	}
}
void createSpheres(){
	int i = 0;
	for (int j=0; j<SPHERES; j++) {
		if(busy_array[j] == 0)
		{
			i = j;
			break;
		}
	}

	temp_s[i].r = rnd(1.0f);
	temp_s[i].g = rnd(1.0f);
	temp_s[i].b = rnd(1.0f);
	temp_s[i].radius = rnd(80.0f) + 5;
	temp_s[i].x = -width/2 - temp_s[i].radius; // start at right edge of screen
	temp_s[i].y = -height/2 + 2*temp_s[i].radius + rnd(height);				// start at random location
	temp_s[i].z = i;//z_location;				//
	temp_s[i].V_x = rnd(450.f) + INITIAL_X_SPEED;	// initial x-speed
	temp_s[i].V_y = INITIAL_Y_SPEED;
	busy_array[i] = 1;
}
void keyboard(unsigned char key, int x, int y)
{
		switch (key) 
		{
			case 'a': 
			createSpheres(); 
			break;
			case 'b': 
			for(int k = 0; k < 5; k++){
					createSpheres();
				}
			break;
			case 'c': 
				for(int k = 0; k < 10; k++){
					createSpheres();
				}
			break;	
		}
}
void moveSpheres(int frameTime) {
	for (int i=0; i<SPHERES; i++) {
		temp_s[i].x += temp_s[i].V_x*frameTime/100;
		temp_s[i].y += temp_s[i].V_y*frameTime/100;
		temp_s[i].V_y -= GRAVITY;
		if (temp_s[i].x > temp_s[i].radius + width/2) { busy_array[i] = 0; }
		if (temp_s[i].y + temp_s[i].radius > height/2) { temp_s[i].y = height/2 - temp_s[i].radius; temp_s[i].V_y = -temp_s[i].V_y;}
		if (temp_s[i].y - temp_s[i].radius < -height/2) { temp_s[i].y = temp_s[i].radius - height/2; temp_s[i].V_y = -BOUNCE_COEFFICIENT*temp_s[i].V_y;}
	}
}
void RenderScene(void)
{

	cudaError_t cudaStatus;
	
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);
	dim3 grids(width/BLOCK_SIZE,height/BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE);

	switch(mode)
	{
		case 0: 
			cudaStatus = cudaMemcpy(s, temp_s, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice);
			kernel<<<grids,threads>>>(dev_bitmap, s, width, height);
			cudaMemcpy( cpu_bitmap, dev_bitmap,4*sizeof(char)*width*height, cudaMemcpyDeviceToHost);
			break;
		case 1:
			cudaStatus = cudaMemcpyToSymbol((const char*)c_s, temp_s, sizeof(Sphere)*SPHERES);
			kernel_c<<<grids,threads>>>(dev_bitmap, width, height);
			cudaMemcpy( cpu_bitmap, dev_bitmap,4*sizeof(char)*width*height, cudaMemcpyDeviceToHost);
			break;
		case 2:
			cudaStatus = cudaMemcpyToSymbol((const char*)c_s, temp_s, sizeof(Sphere)*SPHERES);
			cudaStatus = cudastream();
			break;
	}
	
	cudaEventRecord( stop, 0);
	cudaEventSynchronize(stop);
	float frameTime;
	cudaEventElapsedTime(&frameTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawPixels(width,height,GL_RGBA,GL_UNSIGNED_BYTE,cpu_bitmap);

	glutSwapBuffers();

	glutPostRedisplay();
	iterations++;
	if (iterations < 1000) elapsedTime += frameTime;
	moveSpheres(frameTime);
	//printf("FT:%g", frameTime);
}
void ChangeSize(int w, int h)
{
	cudaError_t cudaStatus; 
	width = w/32 * 32;
	height = h/32 * 32;
	cudaFree(dev_bitmap);
	free(cpu_bitmap);
	cpu_bitmap = (unsigned char*)malloc(4*sizeof(char)*width*height);
	cudaStatus = cudaMalloc((void**)&dev_bitmap, 4*sizeof(char)*width*height);
	glViewport(0,0,w,h);
}
int main(int argc, char* argv[])
{
	cudaError_t cudaStatus; 
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// mode = 0 is global memory version
	// mode = 1 is constant memory version no stream
	// mode = 2 is the stream version using the constant memory
		mode = 1;

		elapsedTime = 0.0f;
		iterations = 0;
		width = 1024;	// set width of window
		height = 768;	// set height of window
		temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);	// malloc temp_s
		//createSpheres();										// generate spheres at temp_s
		cpu_bitmap = (unsigned char*)malloc(4*sizeof(char)*width*height);	// malloc 4 bytes per pixel for cpu
		cudaStatus = cudaMalloc((void**)&dev_bitmap, 4*sizeof(char)*width*height); // malloc dev_bitmap on GPU
		cudaStatus = cudaMalloc((void**)&s, sizeof(Sphere)*SPHERES);

		glutInit(&argc,argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
		glutInitWindowSize(width, height);
		glutCreateWindow("Ray Tracing");
		
		//We now set our event handlers. Again, super easy because GLUT intercepts all of the windows messages and passes them to the appropriate function.
		glutReshapeFunc(ChangeSize);
	//	glutSpecialFunc(SpecialKeys);
		glutDisplayFunc(RenderScene);
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION);
	
		glutKeyboardFunc(keyboard);

		//We now enter the windows main loop.. The render scene function will now be called continuously.
		glutMainLoop();

		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			goto Error;
		}
		switch (mode)
		{
			case 0:	printf("----Using Global Memory for Spheres----\n");	break;
			case 1:	printf("----Using Constant Memory for Spheres----\n");	break;
			case 2:	printf("----Using Streaming Memory for Spheres----\n");	break;
		}
		printf("Average CUDA Processing Time = %5f ms\n",elapsedTime/1000);

	char x;
	std::cin >> x;
Error:
	free(temp_s);
	cudaFree(dev_bitmap);
	cudaFree(s);
	free(cpu_bitmap);
    return 0;
}
cudaError_t cudastream()
{
	cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);

	int effective_height;
	effective_height = height/(2*NUM_CHUNKS);

	dim3 grids0(width/BLOCK_SIZE,effective_height/BLOCK_SIZE);
	dim3 threads0(BLOCK_SIZE,BLOCK_SIZE);

	unsigned char* dev_bitmap0;
	unsigned char* dev_bitmap1;
	
	cudaStatus = cudaMalloc((void**)&dev_bitmap0, 4*sizeof(char)*width*effective_height);
	cudaStatus = cudaMalloc((void**)&dev_bitmap1, 4*sizeof(char)*width*effective_height);
	
	//initialize the stream

	cudaStream_t stream, stream1;
	cudaStatus = cudaStreamCreate(&stream);
	cudaStatus = cudaStreamCreate(&stream1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaStreamCreate failed!");
		goto Error;
	}
	for(int i = 0; i<NUM_CHUNKS; i++){

		kernel_cs<<<grids0,threads0, 0, stream>>>(dev_bitmap0, width, height, 2*i*effective_height);
		kernel_cs<<<grids0,threads0, 0, stream1>>>(dev_bitmap1, width, height, (2*i+1)*effective_height);
	
		cudaStatus = cudaMemcpyAsync(cpu_bitmap+2*i*4*width*effective_height, dev_bitmap0, 4*width*sizeof(char)*effective_height, cudaMemcpyDeviceToHost, stream);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpyAsync failed!");
			goto Error;
		}
	
		cudaStatus = cudaMemcpyAsync(cpu_bitmap+(2*i+1)*4*width*effective_height, dev_bitmap1, 4*width*sizeof(char)*effective_height, cudaMemcpyDeviceToHost, stream);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpyAsync failed!");
			goto Error;
		}
	}
	
	cudaStatus = cudaStreamSynchronize(stream);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaStreamSync failed!");
		goto Error;
	}
	cudaStatus = cudaStreamSynchronize(stream1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaStreamSync failed!");
		goto Error;
	}


Error:
	cudaFree(dev_bitmap0);
	cudaFree(dev_bitmap1);
	//cudaFreeHost(Pcpu_bitmap);
	cudaStreamDestroy(stream);
	cudaStreamDestroy(stream1);
	return cudaStatus;
}