#include "caffe/common.hpp"
#include "caffe/security.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include <string>
#include <fstream>
using namespace caffe;
DEFINE_string(proto, "deploy.prototxt",
    "The defination of caffe model.");

DEFINE_string(model, "VGG.caffemodel",
    "The weights of caffe model.");

DEFINE_string(path, "./",
    "Save path.");

void encrypt_file(Security* security, const std::string& file, const std::string & save ){
    std::ifstream ifs(file.c_str());
    std::ofstream ofs(save.c_str());
    if(ifs.is_open()){
        std::string content =std::string((std::istreambuf_iterator<char>(ifs) ),
                       (std::istreambuf_iterator<char>()) );
        ifs.close();
        std::string content_encrypt;
        
        security->encrypt(content, content_encrypt);
        LOG(INFO)<<"read " << content.size() <<" bytes" <<" after encrypt size: "<< content_encrypt.size();
        if(ofs.good()){
            ofs.write(content_encrypt.c_str(),content_encrypt.length());
            //ofs << content_encrypt;
            LOG(INFO)<<"write "<<content_encrypt.size() <<" bytes";
            ofs.close();
        }else{
            LOG(INFO)<<"can not save "<< save;
        }
    }else{
        LOG(INFO)<<"can not find "<< file;

    }
}

int main(int argc, char** argv){
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;
    gflags::SetUsageMessage("encrypt caffe model including proto and *.caffemodel.\n"
        "Usage:\n"
        "    encrypt_model [FLAGS] ROOTFOLDER\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc != 2) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/encrypt_model");
        return 1;
    }
    string key = argv[1];
    Security* security = new Security(key);
    encrypt_file(security, FLAGS_proto, FLAGS_path + "/model_txt");
    encrypt_file(security, FLAGS_model, FLAGS_path + "/model_binary");

    delete security;



}