# PETcat_vision  <br/>
##Some Info:  <br/>
1.Use rbuild for building<br/>  
2./at contains all folders(/s'i' for i in classes) with each folder having 10 pictures for training  <br/>
3.Use at.txt with ./face exec as: ./face at.txt  <br/>
4.at.txt contains paths to all images with label, helping in loading dataset<br/>  
5./src contains python scripts for polishing images and generating csv though you won't need them<br/>  
6./src/face_recog.cpp is the file you use also note /src contains the haarcascade xml and ignore tut.cpp file<br/>  
7.Cheers!  <br/>

_________________________________________________
Vision modules for PETCat, the long term project.

## Contribution Guidelines

### 1. Fork the repo on GitHub: Use the button at the top right.

### 2. Clone the project to your own machine

``` git clone https://github.com/${your_username_here}/PETcat_vision ```

### 3. Commit changes to your own branch: 

Create a new branch by

``` git checkout -b ${your_branch_name} ```

### 4. Push your work back up to your fork: 

Navigate to the top-level repo directory and:
``` 
git add .
git commit -m "Explanative commit message"
git push origin ${your_branch_name} 
```
    
### 5. Submit a Pull request so that we can review your changes:

Create a new pull request from the `Pull Requests` tab on this repo, not the fork.

Request reviews from at least two people. 
  
