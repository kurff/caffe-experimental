#ifndef __BOX_H__
#define __BOX_H__
#include <string>
namespace rcnn{
    typedef struct Box_{
        float x;
        float y;
        float width;
        float height;
        int label;
        std::string label_name;
        float score;
    }Box;


}
#endif