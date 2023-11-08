
import numpy as np
from scipy.interpolate import interp2d


# helper functions
def get_rot_x(angle):
    '''
    transformation matrix that rotates a point about the standard X axis
    '''
    Rx = np.zeros(shape=(3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(angle)
    Rx[1, 2] = -np.sin(angle)
    Rx[2, 1] = np.sin(angle)
    Rx[2, 2] = np.cos(angle)
    
    return Rx

def get_rot_y(angle):
    '''
    transformation matrix that rotates a point about the standard Y axis
    '''
    Ry = np.zeros(shape=(3, 3))
    Ry[0, 0] = np.cos(angle)
    Ry[0, 2] = -np.sin(angle)
    Ry[2, 0] = np.sin(angle)
    Ry[2, 2] = np.cos(angle)
    Ry[1, 1] = 1
    
    return Ry

def get_rot_z(angle):
    '''
    transformation matrix that rotates a point about the standard Z axis
    '''
    Rz = np.zeros(shape=(3, 3))
    Rz[0, 0] = np.cos(angle)
    Rz[0, 1] = -np.sin(angle)
    Rz[1, 0] = np.sin(angle)
    Rz[1, 1] = np.cos(angle)
    Rz[2, 2] = 1
    
    return Rz

def create_V2I_Mat(focalLength,principalPoint,height,pitch, yaw, roll):
    Pitch = pitch * np.pi / 180 # pitch angle in radians
    Yaw = yaw * np.pi / 180 # yaw angle in radians
    Roll = roll * np.pi / 180 # roll angle in radians
    rotationMatrix=np.matmul(get_rot_z(-Yaw),get_rot_x(np.pi/2-Pitch),get_rot_z(Roll))
    translationInWorldUnits = np.array([0,0, height]) # translation vector in world units
    translation = np.matmul(translationInWorldUnits, rotationMatrix)
    rotation = np.matmul(get_rot_y(np.pi),get_rot_z(-np.pi/2),get_rot_z(-Yaw))
    rotation = np.matmul(rotation ,get_rot_x(np.pi/2-Pitch),get_rot_z(Roll))
    camMatrix = np.row_stack((rotation,translation))
    intrinsicMatrix = np.array([[focalLength[0], 0, principalPoint[0]],
                            [0, focalLength[1], principalPoint[1]],
                            [0, 0, 1]]) 
    camMatrix =np.matmul(camMatrix,intrinsicMatrix.T)
    V2I2D = np.delete(camMatrix,2,axis=0)
    return V2I2D

def create_V2B_Mat(distAheadOfSensor,spaceToOneSide,bottomOffset,reqImgWidth,I2V2D,imageSize):
    outView   = np.array([bottomOffset, distAheadOfSensor, -spaceToOneSide, spaceToOneSide]); # [xmin, xmax, ymin, ymax] xmin对应bottomOffset outView=[3,30,-6,+6]
    worldHW = abs(outView[[1,3]] - outView[[0,2]]) # [27,12]
    scale = (reqImgWidth-1) / worldHW[1] # 249/12=20.75
    scaleXY = np.array([scale, scale]) # [20.75,20.75]
    reqHeight = round(worldHW[0] * scaleXY[0])+1 # 27*20.75=560.25
    outSize = np.array([reqHeight , reqImgWidth]) # [561,250]
    dYdXVehicle = np.array([outView[3] , outView[1]]) # [561,250]
    tXY = scaleXY*dYdXVehicle # [124.5 622.5]
    viewMatrix = np.array([[scaleXY[0], 0, 0],
                            [0, scaleXY[1], 0],
                            [tXY[0]+1, tXY[1]+1, 1]]) 
    adjTform = np.array([[0, -1, 0],#-1 here causes y to become x and reverses the axis direction
                        [-1, 0, 0],#-1 here causes x to become y and reverses the axis direction
                            [0, 0, 1]]) # no translation
    bevTform = np.matmul(I2V2D, adjTform)
    tform = np.matmul(bevTform,viewMatrix)
    formt = np.linalg.inv(tform)
    R_A_XWorldLimits = np.array([0.5, imageSize[1]+0.5]) # 0.5,640.5
    R_A_YWorldLimits = np.array([0.5, imageSize[0]+0.5]) # 0.5,480.5
    R_A_PixelExtentInWorldX = 1
    R_A_PixelExtentInWorldY = 1
    R_A_ImageExtentInWorldX = imageSize[1]# 640
    R_A_ImageExtentInWorldY = imageSize[0]# 480
    R_A_XIntrinsicLimits = np.array([0.5, imageSize[1]+0.5])# 0.5,640.5
    R_A_YIntrinsicLimits = np.array([0.5, imageSize[0]+0.5])# 0.5,480.5
    outputRef_XWorldLimits = np.array([0.5,outSize[1]+0.5]) # 0.5,640.5
    outputRef_YWorldLimits = np.array([0.5,outSize[0]+0.5]) # 0.5,480.5
    outputRef_PixelExtentInWorldX = 1
    outputRef_PixelExtentInWorldY = 1
    outputRef_ImageExtentInWorldX =outSize[1]# 640
    outputRef_ImageExtentInWorldY =outSize[0]# 480
    outputRef_XIntrinsicLimits = np.array([0.5,outSize[1]+0.5])# 0.5,640.5
    outputRef_YIntrinsicLimits = np.array([0.5,outSize[0]+0.5])# 0.5,480.5
    [dstXIntrinsic,dstYIntrinsic] = np.meshgrid(np.arange(0,outSize[1]),np.arange(0,outSize[0]))
    Sx = outputRef_PixelExtentInWorldX
    Sy = outputRef_PixelExtentInWorldY
    Tx = outputRef_XWorldLimits[0] - outputRef_PixelExtentInWorldX * (outputRef_XIntrinsicLimits[0])
    Ty = outputRef_YWorldLimits[0] - outputRef_PixelExtentInWorldY * (outputRef_YIntrinsicLimits[0])
    tIntrinsictoWorldOutput = np.array([[Sx, 0, 0],
                                    [0, Sy, 0],
                                    [Tx, Ty, 1]])
    Sx = 1/R_A_PixelExtentInWorldX
    Sy = 1/R_A_PixelExtentInWorldY
    Tx = R_A_XWorldLimits[0] - 1/R_A_PixelExtentInWorldX * (R_A_XIntrinsicLimits[0])
    Ty = R_A_YWorldLimits[0] - 1/R_A_PixelExtentInWorldY * (R_A_YIntrinsicLimits[0])
    tWorldToIntrinsicInput = np.array([[Sx, 0, 0],
                                    [0, Sy, 0],
                                    [Tx, Ty, 1]])
    tComp = np.matmul(tIntrinsictoWorldOutput,formt,tWorldToIntrinsicInput)
    tformComposite = tComp.T
    B = tformComposite
    alpha = np.ones_like(dstXIntrinsic)
    up = B[0,0]*dstXIntrinsic + B[0,1]*dstYIntrinsic + B[0,2]*alpha
    vp = B[1,0]*dstXIntrinsic + B[1,1]*dstYIntrinsic + B[1,2]*alpha
    beta = B[2,0]*dstXIntrinsic + B[2,1]*dstYIntrinsic + B[2,2]*alpha
    srcXIntrinsic = up/beta
    srcYIntrinsic = vp/beta
    tform2 = np.matmul(tComp,I2V2D)
    tform2inv = np.linalg.inv(tform2)
    return outSize,tform2inv,tform2,srcXIntrinsic,srcYIntrinsic

def compute_uv2xy_projection(uv_points, M, is_homogeneous=False):
    '''
    Given a set of points in the uv coordinate system and the overall tran-matrix,
    compute the projection of uv points onto the xy point
    
    Parameters
    -----------
    uv_points - np.ndarray, shape - (2, n_points)
                   points in the uv coordinate system
                   
    M - np.ndarray, shape - (3, 3)
        The overall  tran-matrix
        : 
        V2I_Mat_T = V2I_Mat.T
        I2V_Mat_T = I2V_Mat.T
        V2B_Mat_T = V2B_Mat.T
        B2V_Mat_T = B2V_Mat.T
        I2B_Mat_T = I2B_Mat.T
        B2I_Mat_T = B2I_Mat.T
        
    is_homogeneous - boolean
        whether the coordinates are represented in their homogeneous form
        if False, an extra dimension will  be added for computation
        
    Returns
    ----------
    projections - np.ndarray, shape - (2, n_points)
                  projections of the xy points onto the image
    '''
    if not is_homogeneous:
        # convert to homogeneous coordinates
        points_h = np.vstack((uv_points, np.ones(uv_points.shape[1])))
        
    h_points_i = M @ points_h
    
    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_xy = h_points_i[:2, :]    
    
    return points_xy

def create_birdimage(inputImage, srcXIntrinsic,srcYIntrinsic):
    X = np.arange(0,inputImage.shape[1])
    Y = np.arange(0,inputImage.shape[0])
    f = interp2d(X, Y, inputImage, kind='linear')
    BirdImage = np.ones([srcXIntrinsic.shape[0],srcXIntrinsic.shape[1]])
    Xq = srcXIntrinsic.flatten()    
    Yq = srcYIntrinsic.flatten()
    for i in range(srcXIntrinsic.shape[0]):
        BirdImage[i,:] = f(Xq[i*srcXIntrinsic.shape[1]:(i+1)*srcXIntrinsic.shape[1]], Yq[i])
    return BirdImage

def np2cv(nparray):
    max_n = nparray.max()
    min_n = nparray.min()
    nparrayMat = 255 * (nparray - min_n) / (max_n - min_n)
    nparrayMat= np.asarray(nparrayMat.astype(np.uint8), order="C")
    return nparrayMat