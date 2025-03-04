#ifndef _SCORE_HPP_
#define _SCORE_HPP_

#include <fstream>
#include <vector>

using namespace std;

enum ROLE {
    GOALKEEPER = 1,
    DEFENDER,
    MIDFIELDER,
    FORWARD
};

#define NUM_WEIGHTS 7

// GOALKEEPER
#define GK_WEIGHT_GOALS     0.7
#define GK_WEIGHT_ASSISTS   0.3
#define GK_WEIGHT_XGI       0.1
#define GK_WEIGHT_BP        0.5
#define GK_WEIGHT_ICT       0.0
#define GK_WEIGHT_GC        -0.4
#define GK_WEIGHT_XGC       -0.2

// DEFENDER
#define DEF_WEIGHT_GOALS     0.6
#define DEF_WEIGHT_ASSISTS   0.3
#define DEF_WEIGHT_XGI       0.0
#define DEF_WEIGHT_BP        0.4
#define DEF_WEIGHT_ICT       0.0
#define DEF_WEIGHT_GC        -0.1
#define DEF_WEIGHT_XGC       -0.05

// MIDFIELDER
#define MID_WEIGHT_GOALS     0.5
#define MID_WEIGHT_ASSISTS   0.3
#define MID_WEIGHT_XGI       0.0
#define MID_WEIGHT_BP        0.3
#define MID_WEIGHT_ICT       0.0
#define MID_WEIGHT_GC        -0.05
#define MID_WEIGHT_XGC       0.0

// FORWARD
#define FWD_WEIGHT_GOALS     0.4
#define FWD_WEIGHT_ASSISTS   0.3
#define FWD_WEIGHT_XGI       0.0
#define FWD_WEIGHT_BP        0.3
#define FWD_WEIGHT_ICT       0.0
#define FWD_WEIGHT_GC        0.0
#define FWD_WEIGHT_XGC       0.0


static inline int setWeights(vector<float>& w, ROLE role) {

    switch (role)
    {
    case GOALKEEPER:
        w = {   GK_WEIGHT_GOALS,
                GK_WEIGHT_ASSISTS,
                GK_WEIGHT_XGI,
                GK_WEIGHT_BP,
                GK_WEIGHT_ICT,
                GK_WEIGHT_GC,
                GK_WEIGHT_XGC
        };
        break;
    case DEFENDER:
        w = {   DEF_WEIGHT_GOALS,
                DEF_WEIGHT_ASSISTS,
                DEF_WEIGHT_XGI,
                DEF_WEIGHT_BP,
                DEF_WEIGHT_ICT,
                DEF_WEIGHT_GC,
                DEF_WEIGHT_XGC
        };
        break;
    case MIDFIELDER:
        w = {   MID_WEIGHT_GOALS,
                MID_WEIGHT_ASSISTS,
                MID_WEIGHT_XGI,
                MID_WEIGHT_BP,
                MID_WEIGHT_ICT,
                MID_WEIGHT_GC,
                MID_WEIGHT_XGC
        };
        break;
    case FORWARD:
        w = {   FWD_WEIGHT_GOALS,
                FWD_WEIGHT_ASSISTS,
                FWD_WEIGHT_XGI,
                FWD_WEIGHT_BP,
                FWD_WEIGHT_ICT,
                FWD_WEIGHT_GC,
                FWD_WEIGHT_XGC
        };
        break;
    
    default:
        break;
    }

    return EXIT_SUCCESS;
}


//
float calculateEvaluationScore(
    float goals, float assists, float xG, float xA,
    float bonus_points, float influence, float creativity, float threat,
    float goals_conceded, float xGC, ROLE role
);



#endif //_SCORE_HPP_
