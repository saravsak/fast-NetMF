#include "utils.h"
#include<time.h>
#include<stdlib.h>
#include<iostream>

void log(const char* message){
        time_t rawtime;
	struct tm* timeinfo;

	time(&rawtime);
	timeinfo = localtime (&rawtime);

	char *timetext = asctime(timeinfo);
	timetext[24] = '\0'; 

	std::cout<<"["<<timetext<<"]: "<<message<<std::endl;
}

double diff_ms(timeval t1, timeval t2)
{
    return (((t1.tv_sec - t2.tv_sec) * 1000000) + 
            (t1.tv_usec - t2.tv_usec))/1000;
}
