/************************************************************************
*
*
*   changed by sheng wang
*   2017.10.11
*
*
*  lap.h
   version 1.0 - 21 june 1996
   author  Roy Jonker, MagicLogic Optimization Inc.
   
   header file for LAP
*
**************************************************************************/

/*************** CONSTANTS  *******************/

  #define BIG 100000

/*************** TYPES      *******************/

  typedef int row;
  typedef int col;
  typedef double cost;

/*************** FUNCTIONS  *******************/
#include <vector>
using namespace std;

//using namespace std;
//extern double lap(int dim, double **assigncost,
//               int *rowsol, int *colsol, double *u, double *v);
extern double lap(int dim, std::vector< std::vector<double> >& assigncost,
                  int *rowsol, int *colsol, double *u, double *v);

//extern double lap(int dim, std::vector< std::vector<double> >& assigncost,
//                  vector<int>& rowsol, vector<int>& colsol, vector<double>& u, vector<double>& v);

//extern void checklap(int dim, double **assigncost,
//                     int *rowsol, int *colsol, double *u, double *v);

