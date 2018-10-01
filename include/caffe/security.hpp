#ifndef __SECURITY_HPP__
#define __SECURITY_HPP__
#include <string>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <iostream>
namespace caffe{
    class Security{
        public:
            Security(std::string key):key_(key){
                salt_[0] = 12345;
                salt_[1] = 54321;
                init();
            }

            ~Security(){
                EVP_CIPHER_CTX_cleanup(&en_);
                EVP_CIPHER_CTX_cleanup(&de_);
            }

            int init(){ 
                if (aes_init((unsigned char* )key_.c_str(), key_.size(), (unsigned char *)&salt_, &en_, &de_)) {
                    printf("Couldn't initialize AES cipher\n");
                    return -1;
                }
            }

            void encrypt(const std::string& input, std::string& output){
                int len = input.size();
                unsigned char* o = aes_encrypt(&en_, (unsigned char*)input.c_str(), & len);
                output = std::string(reinterpret_cast<char*>(o), len);

            }

            void decrypt(const std::string& input, std::string& output){
                int len = input.size();
                //std::cout<<  len << std::endl;Security
                unsigned char* o = aes_decrypt(&de_, (unsigned char*) input.c_str(), & len);
                //std::cout<< len << std::endl;
                output = std::string(reinterpret_cast<char*>(o), len);
                //std::cout<< *output <<std::endl;
            }


            /**
                * Create a 256 bit key and IV using the supplied key_data. salt can be added for taste.
                * Fills in the encryption and decryption ctx objects and returns 0 on success
            **/
            int aes_init(const unsigned char *key_data, int key_data_len, unsigned char *salt, EVP_CIPHER_CTX *e_ctx, 
                EVP_CIPHER_CTX *d_ctx)
            {
                int i, nrounds = 5;
                unsigned char key[32], iv[32];
  
                /*
                * Gen key & IV for AES 256 CBC mode. A SHA1 digest is used to hash the supplied key material.
                * nrounds is the number of times the we hash the material. More rounds are more secure but
                * slower.
                */
                i = EVP_BytesToKey(EVP_aes_256_cbc(), EVP_sha1(), salt, key_data, key_data_len, nrounds, key, iv);
                if (i != 32) {
                    printf("Key size is %d bits - should be 256 bits\n", i);
                    return -1;
                }

                EVP_CIPHER_CTX_init(e_ctx);
                EVP_EncryptInit_ex(e_ctx, EVP_aes_256_cbc(), NULL, key, iv);
                EVP_CIPHER_CTX_init(d_ctx);
                EVP_DecryptInit_ex(d_ctx, EVP_aes_256_cbc(), NULL, key, iv);

                return 0;
            }

            /*
            * Encrypt *len bytes of data
            * All data going in & out is considered binary (unsigned char[])
            */
            unsigned char *aes_encrypt(EVP_CIPHER_CTX *e, const unsigned char *plaintext, int *len)
            {
                /* max ciphertext len for a n bytes of plaintext is n + AES_BLOCK_SIZE -1 bytes */
                int c_len = *len + AES_BLOCK_SIZE;
                int f_len = 0;
                unsigned char *ciphertext = (unsigned char*)malloc(c_len);
                /* allows reusing of 'e' for multiple encryption cycles */
                EVP_EncryptInit_ex(e, NULL, NULL, NULL, NULL);

                /* update ciphertext, c_len is filled with the length of ciphertext generated,
                *len is the size of plaintext in bytes */
                EVP_EncryptUpdate(e, ciphertext, &c_len, plaintext, *len);

                /* update ciphertext with the final remaining bytes */
                EVP_EncryptFinal_ex(e, ciphertext+c_len, &f_len);

                *len = c_len + f_len;
                return ciphertext;
            }

            /*
            * Decrypt *len bytes of ciphertext
            */
            unsigned char *aes_decrypt(EVP_CIPHER_CTX *e, const unsigned char *ciphertext, int *len)
            {
                /* plaintext will always be equal to or lesser than length of ciphertext*/
                int p_len = *len, f_len = 0;
                unsigned char *plaintext = (unsigned char*)malloc(p_len);
  
                EVP_DecryptInit_ex(e, NULL, NULL, NULL, NULL);
                EVP_DecryptUpdate(e, plaintext, &p_len, ciphertext, *len);
                EVP_DecryptFinal_ex(e, plaintext+p_len, &f_len);

                *len = p_len + f_len;
                return plaintext;
            }
        protected:
            std::string key_;
            EVP_CIPHER_CTX en_;
            EVP_CIPHER_CTX de_;
            unsigned int salt_[2];

    };

}


#endif