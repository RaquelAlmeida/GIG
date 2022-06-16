
#System packages
import argparse
import sys
import os
from pathlib import Path
from time import time

#Dependencies
import pickle
import higra as hg 
import cv2
import numpy as np
import multiprocessing

#Support functions
from utils import resize, conv_tri, rgb2luv, gradient, histogram


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('\nERROR: %s\n' % message)
        self.print_help()
        sys.exit(2)

def parserFunction():
    '''
    Input Parser function
    '''
    parser=MyParser(description='Compute GIG gradients from input image(s) or Pink weighted graph(s)',add_help=True)
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument("--type", help="Indicate the type of the input as image (I) or weighted graph (G).", choices=['I','G'], required=True)
    requiredNamed.add_argument("--input", help="The path for the input. If the path is a directory the program will compute the gradient for all files within.", required=True)
    parser.add_argument("--output", help="Path to save the computed gradient. Default= ./GIG_gradients/", default='GIG_gradients/')
    parser.add_argument("--model", help="Path to the trained GIG model. Default= ./GIG-model.pkl/", default='GIG-model.pkl')
    args=parser.parse_args()
    print()
    print('Compute GIG gradients for:')
    if args.type == "I": print("Input type: image")
    else: print("Input type: graph")
    print("Input path:", args.input)
    print("Out folder:", args.output)
    print("Model path:",args.model)
    print()
    return args.type, args.input, args.output, args.model

def pinkMessage():
    print()
    print("Pink weighted graph(s). They must contain:")
    print("\t- A 8-adjacency undirected graph;")
    print("\t- A 2D numpy array containing the grayscale representation of the original image; and")
    print("\t- A numpy array of shape [number_of_edges,] containing the edges' weights.")  


def padding(l, n):
    '''
    Padding function
    '''
    return l[:n] + [1]*(n-len(l))


def getColorFeat(src):
    '''
    Support function to compute color descriptors
    '''
    #number of gradient orientation
    n_orientation=4

    #normalization radius for gradient
    radius_norm = 4
    #Convert to lab
    luv = rgb2luv(src)
    size = (luv.shape[0], luv.shape[1])
    channels = [resize(luv, size)]
    reg_smooth_rad = 2

    for scale in [1.0, 0.5]:
        img = resize(luv, (luv.shape[0] * scale, luv.shape[1] * scale))
        magnitude, orientation = gradient(img, radius_norm)
        downscale = max(1, int(1 * scale))
        hist = histogram(magnitude, orientation, downscale, n_orientation)
        channels.append(resize(magnitude, size)[:, :, None])
        channels.append(resize(hist, size))

    channels = np.concatenate(channels, axis=2)            
    
    if reg_smooth_rad > 1.0:
        reg_ch = conv_tri(channels, int(round(reg_smooth_rad)))
    else:
        reg_ch = conv_tri(channels, reg_smooth_rad)

    return reg_ch


def getFeaturesGraph(g,edge_weights,img,size,image_name):
    '''
    Create regular representation and predict output
    '''

    #Path for computed gradient
    output_file=os.path.join(output_path,Path(image_name).stem+'.png')

    #Feature size: 13 dimensions for color descriptors + 64 neighbors
    ftr_size=77
    ftrs=np.zeros((g.num_vertices(),ftr_size))
    
    #Compute color features
    color_features=getColorFeat(img)
    
    # Get edge weights for imediate neighbors and neighbors of neighbors
    for v in g.vertices():
        x,y=np.unravel_index(v, size)
        ftrs[v,0:13]=color_features[x,y,:]
        nei=[]
        for e in g.out_edges(v):
            u=g.target(e)         
            w=edge_weights[g.index(e)]
            nei.append(w)
                
            nei_nei=[]            
            for f in g.out_edges(u):
                p=g.target(f)
                if p==v: continue         
                w=edge_weights[g.index(f)]
                nei_nei.append(w)
                    
            nei_nei=padding(nei_nei,7)
            nei.extend(nei_nei)        
        nei=padding(nei,64)
        ftrs[v,13:ftr_size]=nei
    
    # Get prediction 
    pred=model.predict(ftrs)
    #Convert predictions to gradient and save
    indexes=np.reshape(pred, size)
    rescaled = (255.0 / indexes.max() * (indexes - indexes.min())).astype(np.uint8)
    cv2.imwrite(output_file,rescaled)
    print("GIG gradient saved in: ", output_file)

    return
        
               
    
def getGraph(image_name):   
    '''
    Read image, create graphs and and call function to create regular representation and predictio
    ''' 
    image = cv2.imread(image_name)
    image=image.astype(np.float32)/255
    gradient_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size=gradient_image.shape[:2]
    graph = hg.get_8_adjacency_graph(size)
    edge_weights = hg.weight_graph(graph, gradient_image, hg.WeightFunction.L2_squared)
    getFeaturesGraph(graph,edge_weights,image,size,image_name)    
    
    return 


def getPink(sample):   
    '''
    Read Pink graphs and call function to create regular representation and predictio 
    '''
    graph,vertex_weights,edge_weights=hg.read_graph_pink(sample)
    vertex_weights=np.array(vertex_weights*255,dtype=np.uint8)
    image=cv2.cvtColor(vertex_weights,cv2.COLOR_GRAY2RGB)
    image=image.astype(np.float32)/255
    size=vertex_weights.shape
    getFeaturesGraph(graph,edge_weights,image,size,sample)

    return 
    

    

if __name__ == '__main__':

    #Parse input parameters
    input_type, path,output_path,model_path=parserFunction()

    #Create output folder
    if not os.path.exists(output_path): os.makedirs(output_path)

    #Load model
    model=pickle.load(open(model_path, 'rb'))
    model.n_jobs=1
    
    #Read input path and creat list of files to compute
    if os.path.isfile(path):
        samples=[path]
    elif os.path.isdir(path): 
        samples=[os.path.join(path, file) for file in os.listdir(path)] 
    else:  
        print("\n Please provide a valid input path. Provided: ", path )
        sys.exit()
    
        
    if input_type == "I":
        #Check files
        for sample in samples:             
            image = cv2.imread(sample)
            if image is None:
                print("ERROR: Not a valid input image. Provided: ", sample)
                sys.exit()
        print("\nOverhead time required to convert image to graph.")  
        start_time=time()  
        pool = multiprocessing.Pool()          
        for sample in samples:
            pool.apply_async(getGraph, args=(sample, ))
        pool.close()
        pool.join() 
        print("Time  %s seconds" % (time() - start_time))
    else:
        #Check files
        for sample in samples:
            try:
                graph,vertex_weights,edge_weights=hg.read_graph_pink(sample)                
            except:
                print()
                print("ERROR: Incorrect Pink file. Provided: ", sample)
                sys.exit()
        #Execute GIG for graph(s) in samples
        pinkMessage()       
        pool = multiprocessing.Pool()          
        for sample in samples:
            pool.apply_async(getPink, args=(sample, ))
        pool.close()
        pool.join() 
