# TD3 on Self Driving Car

### State space or Observation space
State space defines the input the TD3 Network (Actor and Critic).
The state_space defined is raw pixels of the cropped Mask image of Citymap.
The above image is obtained by : </br>
**1)** A Isosceles traingle is drawn on the Mask Image representing the car on the map.The center of the traingle is the position of the car in the map.</br>
**2)** To capture orientation of the car ,I chose the method to rotate the triangle (car) about its center, To rotate about its center, Intially a traingle is drawn at position of car with car angle = 0 (Horizontal), then rotate each point of the triangle around its center.</br>
**3)** After the Triangle is drawn ,the image is reduced to one third of its original size , Hence the center of car gets reduced and its position is also updated.</br>
**4)** To avoid failure of cropping at Walls , Padding of 20 pixels around the image is added in white background. 20 pixels padding was chosen as in the kivy env a constraint on wall is provided for 20 pixels.Even here to capture the changes of padding on the center of triangle , center of car is translated by 20 pixels.</br>
---- Image -----</br>
**5)** A 40 x 40 image is cropped and is converted to Grayscale then pixel normalization and at last converted to numpy array which needs to be fed into Actor and Critic Networks.


### Action space

Action space was defined based on the gym environment Box class as it helps in capturing the Continous values and it also provides uniform sampling distribution for Bounded values.</br>
In this scenario , As in TD3 all the actions are applied simulateously in a single step on the agent while training. Hence I decided to take three independent actions as :</br>
1) Rotation of Car :</br>
    The action values provided are [-5 , 5] as it can turn 5 degrees to left or right.I did try to implement a higher value (30,40) for rotation on the car , The car started was just rotating / Circling around few points. 

2) Acceleration : </br>
   The action values provided are [0,1] ,In this scenario a higher value of accleration increases velocity of the car in few timesteps so ,both accleration and velocity are clipped to 1 

3) Deceleration : </br>
    This action was considered as it could represent the braking aspect in car.The action values are [0,0.5] .Here the velocity is clipped to 0 to avoid moving car in backward direction. If for few scenarios when the decelartion value exceeded acceleration for few steps , the velocity would be negative and the car moves in backward .Hence very low range for decelartion

### Rewards 

Rewards considered are Living Penalty , Wall reward , Destination reward.</br>
Wall reward is considered when the car reaches near wall , I had given a Wall reward = -50 and the episode would also end. </br>
Destination reward is taken when car reaches one of the five destinations and reward of 500 is provided.</br>
Living Penalty is -1, which increases the negative rewards in the episode as long as it is on the map.</br>

### Episode
Episode consists of all the observation steps of its own set.</br>
It is ended and done variable is updated to True when :
1) The car reaches wall
2) The car reaches destination
3) After 5000 episode timesteps </br>
After every episode ends the environment is reset and car is randomly intialized.

### Training 

#### Actor : 
  Actor network gets the state_space of env as input in form of (1,40,40 ) grayscale image .Convolution Layers are applied with a stride of two and then a Average pooling on each layer is applied.
  This provides a Feature vector in 1D and a Dense layer is added at last to obtain the Q values for each possible action.

#### Critic :
 Critic Network requires a Multi-Input as it has image as state and and actions from Actor Network ,So we need to combine 2D and 1D data,Here the image is converted into a 1D feature vector after few convolutions then the 1D input data is concatenated with it and forward propagation is finished.

The training part of TD3 is similar to AntBulletEnv , with small changes on env, Instead of Custom Gym env - reset, get_state, step( action) is all implemented inside Game class to be called by update method.The training consists of two part Initial 10K steps by sampling the action_space and then actions according to policy.


# Note :-

To DO :</br>
1) Add orientation, distance from target as state_dimension .

2) Resolve issues on Acceleration and Decelaration in code related to property errors.

3) Add acceleration in y direction (As it might help to extract more information of environments)

4) Calculate the optimum reward values for different rewards (Living Penalty , wall rewards  etc ), from the overall rewards distribution.

5) Explore HyperParameters. 
















