/*
																				Assignment No. 1
																				
Write functions that can model the following:

• Configuration parameters can be taken in for feedforward network
	o Number of hidden layers
	o Number of neurons in every layer
	o Activation units to be used in every layer
	o Network connections among neurons (fully connected or a different N/W architecture)
• Write code for backpropagation algorithm that comes up with NN based model for the above mentioned configuration parameters
• An appropriate technique of avoiding overfitting or regularization to be used. Better still, the choice of the overfitting technique can be made configurable 



Made by

Author: Mugdha Satish Kolhe
Enrollment No: BT17CSE043
Course Code: CSL-450
Course: Machine Learning

================================================================================================================================================================

*/


#include <bits/stdc++.h>
#include <fstream>
#include <string>

#define e 2.73
#define alpha 0.008
#define itr 30
#define FILE "iris.data"
#define output_no 1
#define OUTPUT_ACT 2
#define RATIO 0.7

using namespace std;


vector <vector <vector <float> > > best_weights;
float max(float a, float b)
{
	float sol=b;
	if(a>b)
	{
		sol=a;
	}
	return sol;
}

float _Sigmoid(float x)	//1
{
	float out;
	out=float(1/(1+pow(2.731, -x)));
	return out;
}

float _tanh(float x)	//2
{
	float out;
	out=tanh(x);
	return out;
}

float _ReLu(float x)	//3
{
	float out;
	out=max(0, x);
	return out;
}

//derivative of sigmoid function
float _dSigmoid(float x)
{
	float out;
	out=float(_Sigmoid(x)*(1-_Sigmoid(x)));
	return out;
}

//derivative of tanh function
float _dtanh(float x)
{
	float out;
	out=float(1-(_tanh(x))*(_tanh(x)));
	return out;
}

//derivative of ReLu function
float _dReLu(float x)
{
	float out;
	if(x>0)
	{
		out=1;
	}
	else
	{
		out=0;
	}
	return 0;
}

//helper func to print 2d matrix
void printdata(vector <vector <float > > vect)
{
	cout<<endl<<"PRINTING 2D VECTOR "<<endl;
	for(int i=0; i<vect.size(); i++)
	{
		for(int j=0; j<vect[i].size(); j++)
		{
			cout<<vect[i][j]<<" ";
		}
		cout<<endl;
	}
}

//helper function to print 1d matrix
void printsol(vector <float> vect)
{
	cout<<endl<<"PRINTING 1D VECTOR "<<endl;
	for(int i=0; i<vect.size(); i++)
	{
		cout<<vect[i]<<endl;
	}
}

//helper function to print 3d vector
void display_weight(vector <vector <vector <float > > > weights)
{
	cout<<endl<<"==================================================="<<endl;
    for(int i=0; i<weights.size(); i++)
    {
        cout<<"Between layer "<<i<<" and "<<i+1<<endl;
        for(int j=0; j<weights[i].size(); j++)
        {
            //cout<<"For neuron "<<i<<j<<endl;
            for(int k=0; k<weights[i][j].size(); k++)
            {
                cout<<i<<" "<<j<<" "<<k<<" : "<<weights[i][j][k]<<endl;
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<endl<<"==================================================="<<endl;
}

//function for file reading
vector <vector <float> > read_file()
{
	vector <vector <float> > data;
	ifstream file(FILE);
  	string str;
  	while (std::getline(file, str))
	{
		vector <float > v;
    	string word;
    	int start;

    	// making a string stream
    	stringstream iss(str);

    	// Read and print each word.
    	while (iss >> word)
    	{
    		start=0;
    		int flag=0;
    		float num=0;
    		int j=0;
    		int size=word.length();
    		if(word[0]=='-')
    		{
    			flag=1;
    			start=1;
    			size=1;
			}
			num=0;
    		for(int i=start; i<word.length(); i++)
    		{
    			if(word[i]!='.')
    			{
					float c=word[i]-'0';    					
					num=num*10+c;
    			}
    			else if(word[i]=='.')
    			{
    				j=i+1;
				}				
			}
			int x=size-j;
			float div=pow(10, x);
			if(size!=1)
			{
				num=num/div;				
			}
			if(flag==1)
			{
				num=-num;
			}
			//cout<<num<<" ";
			v.push_back(num);
    		
    	}
    //	cout<<endl;
    	data.push_back(v);
        //cout<<endl;
  	}
	return data;
}

//a unit in neural network which stores some values
class Neuron
{
	public:
		float value;
		vector <int> in;
		vector<int> out;
		int act;
		float in_value;
		float delta;
		//float in_delta;

		Neuron()
		{
			value=0;
			delta=0;
			in_value=0;
			act=2;
		}
};

//function to split input vector given starting point and size
vector <vector <float> > split_input(vector <vector <float> > data, int start, int size, int input_no)
{
	//cout<<endl<<"INPUT"<<endl;
	vector <vector <float > > res;
	float n;
	for(int i=start; i<start+size; i++)
	{
		vector <float> temp;
		for(int j=0; j<input_no; j++)
		{
			n=data[i][j];
			temp.push_back(n);			
		}
		res.push_back(temp);
	}
	//cout<<endl<<"EXITED"<<endl;
	return res;
}

//function to split input vector 1d 
vector <float> split_output(vector <float> data, int start, int size)
{
	//cout<<endl<<"OUTPUT"<<endl;
	vector <float > res;
	float n;
	for(int i=start; i<start+size; i++)
	{
		n=data[i];
		if(n>1)n=1;
		else if(n<-0.99)n=-1;
		else n=0;
		res.push_back(n);
	}
	return res;
}

class NeuralNetwork
{
	public:
		int layers;
		vector <vector <class Neuron> > lay;
		vector <vector <vector <float> > > weights;
		vector <vector <vector <float> > > dw;
		//vector <vector <float> > bias;
		
		void fully_connected(int input, int hid_layers, vector <int> neurons, vector<int> activation);
		void my_connections(int hid_layers, vector <vector <vector <float> > > self, vector <int> activation, int input, vector <int> neurons);
		void test(vector <vector <float > > data, vector <float> sol);
		void display_neuron();
		void weight_creation(int input, vector<int> neurons);
		void initialize_weights();
		void train(vector <vector <float > > data, vector <float> sol, int it, float fact, float decay, float th);
		int train_crossval(vector <vector <float > > data, vector <float> sol, vector <vector <float > > valdata, vector <float> valsol, float fact, float decay, float threshold);

};

void NeuralNetwork::initialize_weights()
{
	for(int i=0; i<weights.size(); i++)
	{
		for(int j=0; j<weights[i].size(); j++)
		{
			for(int k=0; k<weights[i][j].size(); k++)
			{
				//cout<<j<<k;
				float r= (float) rand()/RAND_MAX;	//initialization of weights with small random nos btw -0.5 to 0.5
				r-=0.5;
				weights[i][j][k]=r;						
			}
			
		}
	}
	/*for(int i=0; i<layers-1; i++)
	{
		float r= (float) rand()/RAND_MAX;	//initialization of weights with small random nos btw -0.5 to 0.5
		r-=0.5;
		bias[i]=r;
	}
	display_weight(weights);*/
}
	
void NeuralNetwork::weight_creation(int input, vector<int> neurons)
{
	//cout<<"Weight Creation: "<<endl;

	//storing total no of neurons in each layer
	vector <int> total_layers;
	total_layers.push_back(input);
	for(int i=0; i<neurons.size(); i++)
	{
		total_layers.push_back(neurons[i]);
	}
	total_layers.push_back(output_no);

	//weight and delta weight initialization and creation
	for(int i=0; i<layers-1; i++)
	{
		//cout<<"==============Layer========="<<i<<"================"<<endl<<endl;
		vector < vector <float > > ww;
		vector < vector <float > > dx;
		for(int j=0; j<total_layers[i]; j++)
		{
			//lay[i][j] outgoing to k
			vector <float> w;
			vector <float> d;
			for(int k=0; k<lay[i][j].out.size(); k++)
			{
				//cout<<j<<k;
				float r= (float) rand()/RAND_MAX;	//initialization of weights with small random nos btw -0.5 to 0.5
				r-=0.5;
				w.push_back(r);
				d.push_back(0);						
			}
			ww.push_back(w);
			dx.push_back(d);
		}
		weights.push_back(ww);
		dw.push_back(dx);
	}
	for(int i=0; i<layers-1; i++)
	{
		float r= (float) rand()/RAND_MAX;	//initialization of weights with small random nos btw -0.5 to 0.5
		r-=0.5;
		//bias.push_back(r);
	}
	cout<<endl<<"FUNCTION WEIGHTS EXITED"<<endl<<endl;
	display_weight(weights);
	//display_weight(dw);
}
		
void NeuralNetwork::display_neuron()
{
	for(int i=0; i<lay.size(); i++)
	{
		for(int j=0; j<lay[i].size(); j++)
		{
			cout<<i<<j<<": ";
			cout<<"  inval: "<<lay[i][j].in_value<<endl;
			cout<<"  val: "<<lay[i][j].value<<endl;
			cout<<"  act_func: "<<lay[i][j].act<<endl;
			cout<<"  delta: "<<lay[i][j].delta<<endl;
		}
		cout<<endl;
	}
		
}

void NeuralNetwork::fully_connected(int input, int hid_layers, vector <int> neurons, vector<int> activation)
{
	layers=hid_layers+2;

	//input
	vector <class Neuron> vect;
	for(int i=0; i<input; i++)
	{
		Neuron n;
		//cout<<"Layer 0"<<i<<endl;
		for(int j=0; j<neurons[0]; j++)
		{
			n.out.push_back(j);
		}
		vect.push_back(n);	
	}
	lay.push_back(vect);

	//hidden
	
	
	if(hid_layers==1)
	{
		vector <class Neuron> vect;
		for(int j=0; j<input; j++)
		{
			Neuron n;
			n.act=activation[0];
			for(int k=0; k<input; k++)
			{	
				n.in.push_back(k);
			}
			vect.push_back(n);
		}
		lay.push_back(vect);
	}
		
	for(int i=0; i<hid_layers && hid_layers>1; i++)
	{				
		//For layer i
		vector <class Neuron> vect;
		for(int j=0; j<neurons[i]; j++)
		{
			//cout<<"Layer "<<i+1<<j<<endl;				
			Neuron n;
			n.act=activation[i];

			if(i==0 )	//first hidden layer
			{
				for(int k=0; k<input; k++)
				{	
					n.in.push_back(k);
				}
				for(int k=0; k<neurons[i+1]; k++)
				{
					n.out.push_back(k);
				}
			}
			else if(i==hid_layers-1)	//last hidden layer
			{
				for(int k=0; k<neurons[hid_layers-2]; k++)
				{
					n.in.push_back(k);							
				}
				for(int k=0; k<output_no; k++)
				{
					n.out.push_back(k);
				}
			}
			else	//middle layers
			{
				for(int k=0; k<neurons[i-1]; k++)
				{
					n.in.push_back(k);
				}
				for(int k=0; k<neurons[i+1]; k++)
				{
					n.out.push_back(k);
				}
			}
			vect.push_back(n);
		}
		lay.push_back(vect);	
	}
	//out
	vector <class Neuron> v;
	for(int i=0; i<output_no; i++)
	{
		//cout<<"Layer "<<hid_layers+1<<i<<endl;
		Neuron n;
		n.act=OUTPUT_ACT;
		for(int j=0; j<neurons[hid_layers-1]; j++)
		{
			n.in.push_back(j);
		}
		v.push_back(n);
	}
	lay.push_back(v);
	//display_neuron();
	cout<<endl<<"FUNCTION FULLY CONNECTED EXECUTED"<<endl;

} 

void NeuralNetwork::my_connections(int hid_layers, vector <vector <vector <float> > > self, vector <int> activation, int input, vector<int> neurons)
{
	layers=hid_layers+2;

	//input
	vector <class Neuron> vect;
	for(int i=0; i<input; i++)
	{
		Neuron n;
		//cout<<"Layer 0"<<i<<endl;
		for(int j=0; j<neurons[0]; j++)
		{
			n.out.push_back(j);
		}
		vect.push_back(n);	
	}
	lay.push_back(vect);

	//hidden
	if(hid_layers==1)
	{
		vector <class Neuron> vect;
		for(int j=0; j<input; j++)
		{
			Neuron n;
			n.act=activation[0];
			for(int k=0; k<input; k++)
			{	
				n.in.push_back(k);
			}
			vect.push_back(n);
		}
		lay.push_back(vect);
	}
	
	for(int i=0; i<hid_layers && hid_layers>1; i++)
	{				
		//For layer i
		vector <class Neuron> vect;
		for(int j=0; j<neurons[i]; j++)
		{
			//cout<<"Layer "<<i+1<<j<<endl;				
			Neuron n;
			n.act=activation[i];

			if(i==0)	//first hidden layer
			{
				for(int k=0; k<input; k++)
				{	
					n.in.push_back(k);
				}
				for(int k=0; k<neurons[i+1]; k++)	//change
				{
					if(self[i][j][k]==1)
					{
						n.out.push_back(k);
					}
				}
			}
			else if(i==hid_layers-1)	//last hidden layer
			{
				for(int k=0; k<neurons[hid_layers-2]; k++)	//change
				{
					if(self[i-1][k][j]==1)
					{
						//cout<<"a"<<endl;
						n.in.push_back(k);		
					}
				}
				for(int k=0; k<output_no; k++)
				{
					//cout<<"HELLO MER"<<endl;				
					n.out.push_back(k);
				}
			}
			else	//middle layers	//change
			{
				for(int k=0; k<neurons[i-1]; k++)
				{
					if(self[i-1][k][j]==1)
					{
						//cout<<"b"<<endl;
						n.in.push_back(k);
					}
				}
				for(int k=0; k<neurons[i+1]; k++)
				{
					if(self[i][j][k]==1)
					{
						//cout<<"c"<<endl;
						n.out.push_back(k);
					}
				}
			}
			vect.push_back(n);
		}
		lay.push_back(vect);	
	}
	
	cout<<"output"<<endl;
	//out
	vector <class Neuron> v;
	for(int i=0; i<output_no; i++)
	{
		//cout<<"Layer "<<hid_layers+1<<i<<endl;
		Neuron n;
		n.act=OUTPUT_ACT;
		for(int j=0; j<neurons[hid_layers-1]; j++)
		{
			n.in.push_back(j);
		}
		v.push_back(n);
	}
	for(int i=0; i<layers-1; i++)
	{
		float r= (float) rand()/RAND_MAX;	//initialization of weights with small random nos btw -0.5 to 0.5
		r-=0.5;
		//bias.push_back(r);
	}
	lay.push_back(v);
	display_neuron();
	//cout<<endl<<"FUNCTION SELF CONNECTED EXECUTED"<<endl;

} 

void NeuralNetwork::train(vector <vector <float > > data, vector <float> sol, int it, float fact, float decay, float th)
{
	float min=0, err=0, sum;
	cout<<endl<<"============================================="<<endl;
	cout<<"Training "<<endl;
	cout<<endl<<"============================================="<<endl;
			
	for(int i=0; i<it && th==0 || err<th; i++)
	{
		sum=0;
		err=0;
		cout<<"Epoch "<<i+1<<endl;
		//display_weight(weights);
		//display_dweights();				
				
		//feed forward
		for(int j=0; j<data.size(); j++)
		{
			//for one input row
			//layer by layer calculation of values
			for(int k=0; k<lay.size(); k++)
			{
				for(int l=0; l<lay[k].size(); l++)
				{
					lay[k][l].value=0;
					lay[k][l].in_value=0;
					lay[k][l].delta=0;
				}
			}
			for(int k=0; k<data[j].size(); k++)
			{
				lay[0][k].in_value=data[j][k];
				lay[0][k].value=data[j][k];
				//cout<<data[j][k]<<" ";
			}
			//cout<<endl;
			for(int k=1; k<layers; k++)
			{
				//find connections between them for layer k
				for(int m=0; m<weights[k-1].size(); m++)
				{
					for(int n=0; n<weights[k-1][m].size(); n++)
					{
						//neuron
						lay[k][n].in_value+=weights[k-1][m][n]*lay[k-1][m].value;								
					}							
				}
				//lay[k][n].in_value+=bias[k];
				for(int m=0; m<lay[k].size(); m++)
				{
					//cout<<"hello"<<k<<m<<" "<<lay[k][m].in_value<<endl;
					int act=lay[k][m].act;
					float val;
					if(act==1)
					{
						val=_Sigmoid(lay[k][m].in_value);
					}
					else if(act==2)
					{		
						val=_tanh(lay[k][m].in_value);
					}
					else
					{						
						val=_ReLu(lay[k][m].in_value);
					}
					lay[k][m].value=val;
							
					//cout<<"hello"<<k<<m<<" "<<lay[k][m].value<<endl;							
				}
	
												
			}
			//display_neuron();
					
					
			//backpropogation	
					
			//output
			for(int k=0; k<output_no; k++)
			{
				//acc to act func
				//cout<<sol[j][k]<<endl;
				if(lay[layers-1][k].act==1)
				{
					lay[layers-1][k].delta=(sol[j]-lay[layers-1][k].value)*_dSigmoid(lay[layers-1][k].in_value);
				}
				else if(lay[layers-1][k].act==2)
				{
					lay[layers-1][k].delta=(sol[j]-lay[layers-1][k].value)*_dtanh(lay[layers-1][k].in_value);
				}
				else
				{
					lay[layers-1][k].delta=(sol[j]-lay[layers-1][k].value)*_dReLu(lay[layers-1][k].in_value);
				}
											
			}
			//change in weights for output layer
			for(int k=0; k<weights[layers-2].size(); k++)
			{
				for(int l=0; l<weights[layers-2][k].size(); l++)
				{
					dw[layers-2][k][l]=float(alpha*lay[layers-1][l].delta*lay[layers-1][l].value);
					//cout<<dw[layers-2][k][l]<<endl;
				}
			}
					
			//hidden layer
			//calculation of deltas and weights
			for(int k=layers-2; k>0; k--)
			{
				for(int l=0; l<lay[k].size(); l++)
				{
					float sum=0, val=0;
					for(int m=0; m<weights[k].size(); m++)
					{
						for(int n=0; n<weights[k][m].size(); n++)
						{
							sum+=lay[k+1][n].delta*weights[k][m][n];
						}
					}
					if(lay[k][l].act==1)
					{
						val=_dSigmoid(lay[k][l].in_value);
					}
					else if(lay[k][l].act==2)
					{
						val=_tanh(lay[k][l].in_value);
					}
					else
					{
						val=_dReLu(lay[k][l].in_value);							
					}
					lay[k][l].delta=sum+val;
					//cout<<lay[k][l].delta<<endl;
				}
			}
			for(int k=layers-2; k>=0; k--)
			{
				for(int l=0; l<dw[k].size(); l++)
				{
					for(int m=0; m<dw[k][l].size(); m++)
					{
						dw[k][l][m]=0;;
					}					
				}
			}
			
			//setting values of dw		
			for(int k=layers-2; k>=0; k--)
			{
				for(int l=0; l<dw[k].size(); l++)
				{
					for(int m=0; m<dw[k][l].size(); m++)
					{
						float temp=dw[k][l][m];
						dw[k][l][m]=alpha*lay[k+1][m].delta*lay[k][l].value-temp*fact;
					}					
				}
			}
					
			//updation of weights
			for(int k=0; k<weights.size(); k++)
			{
				for(int l=0; l<weights[k].size(); l++)
				{
					for(int m=0; m<weights[k][l].size(); m++)
					{
						weights[k][l][m]=weights[k][l][m]-dw[k][l][m] + decay*weights[k][l][m];
					}
				}
			}
			/*bias updation
			for(int k=0; k<layers-1; k++)
			{
				bias[k]-=alpha*delta[j];
			}
			*/
			sum+=(lay[layers-1][0].value-sol[j])*(lay[layers-1][0].value-sol[j]);
		//	cout<<"Sum: "<<lay[layers-1][0].in_value<<" "<<sol[j]<<endl;													
		}//one data entry loop ends
		for(int k=0; k<weights.size(); k++)
			{
				for(int l=0; l<weights[k].size(); l++)
				{
					for(int m=0; m<weights[k][l].size(); m++)
					{
						if(weights[k][l][m]>10)	//normalize
							weights[k][l][m]=weights[k][l][m]-10;
					}
				}
			}
		display_weight(weights);
		//display_weight(dw);			
		err=sum/2;
		cout<<endl<<"Error: "<<err<<endl;
			
	}//iterations loop ends
}//func ends

//function to train
int NeuralNetwork::train_crossval(vector <vector <float > > data, vector <float> sol, vector <vector <float > > valdata, vector <float> valsol, float fact, float decay, float threshold)
{
	int ret=0;
	float min=0, err=0, sum=0;
	vector <vector <vector <float> > > best_weights;
	
	//initializing values of best weights
	for(int i=0; i<weights.size(); i++)
	{
		vector < vector <float > > ww;
		for(int j=0; j<weights[i].size(); j++)
		{
			vector <float> w;
			for(int k=0; k<weights[i][j].size(); k++)
			{
				float r= weights[i][j][k];
				w.push_back(r);
									
			}
			ww.push_back(w);
			
		}
		best_weights.push_back(ww);		
	}
	
	//testing on validations set
	for(int j=0; j<valdata.size(); j++)
	{
		//for one input row
		//initialize
		for(int k=0; k<lay.size(); k++)
		{
			for(int l=0; l<lay[k].size(); l++)
			{
				lay[k][l].value=0;
				lay[k][l].in_value=0;
			}
		}
		//layer by layer calculation of values
		for(int k=0; k<valdata[0].size(); k++)
		{
			lay[0][k].in_value=valdata[j][k];
			lay[0][k].value=valdata[j][k];						
			//cout<<data[j][k]<<" ";
		}
		//cout<<endl;
		for(int k=1; k<layers; k++)
		{
			//find connections between them for layer k
			for(int m=0; m<weights[k-1].size(); m++)
			{
				for(int n=0; n<weights[k-1][m].size(); n++)
				{
					//neuron
					lay[k][n].in_value+=best_weights[k-1][m][n]*lay[k-1][m].value;							
				}
			}
			for(int m=0; m<lay[k].size(); m++)
			{
				//cout<<"hello"<<k<<m<<" "<<lay[k][m].in_value<<endl;
				int act=lay[k][m].act;
				float val;
				if(act==1)
				{
					val=_Sigmoid(lay[k][m].in_value);
				}
				else if(act==2)
				{		
					val=_tanh(lay[k][m].in_value);
				}
				else
				{						
					val=_ReLu(lay[k][m].in_value);
				}
				lay[k][m].value=val;
				//cout<<"hello"<<k<<m<<" "<<lay[k][m].value<<endl<<endl;							
			}												
		}
		//calc error
		for(int k=0; k<lay[layers-1].size(); k++)
		{
			
			cout<<lay[layers-1][k].value<<" "<<valsol[j]<<endl;
			sum+=(lay[layers-1][k].value-valsol[j])*(lay[layers-1][k].value-valsol[j]);
		}	
	}
	err=sum/2;
	cout<<"Error z; "<<err<<endl;
	
	cout<<endl<<"============================================="<<endl;
	cout<<"Training "<<endl;
	cout<<endl<<"============================================="<<endl;
	
	float curr_err=err;	
//	curr_err=INT_MAX;	
	while(curr_err-err<threshold && ret<100)
	{
		ret++;
		cout<<"Epoch "<<ret<< " "<<endl;
		//display_weight(weights);
		//display_dweights();				
				
		//feed forward
		for(int j=0; j<data.size(); j++)
		{
			//for one input row
			//layer by layer calculation of values
			for(int k=0; k<lay.size(); k++)
			{
				for(int l=0; l<lay[k].size(); l++)
				{
					lay[k][l].value=0;
					lay[k][l].in_value=0;
					lay[k][l].delta=0;
				}
			}
			for(int k=0; k<data[j].size(); k++)
			{
				lay[0][k].in_value=data[j][k];
				lay[0][k].value=data[j][k];
				//cout<<data[j][k]<<" ";
			}
			//cout<<endl;
			for(int k=1; k<layers; k++)
			{
				//find connections between them for layer k
				for(int m=0; m<weights[k-1].size(); m++)
				{
					for(int n=0; n<weights[k-1][m].size(); n++)
					{
						//neuron
						lay[k][n].in_value+=weights[k-1][m][n]*lay[k-1][m].value;								
					}							
				}
				for(int m=0; m<lay[k].size(); m++)
				{
					//cout<<"hello"<<k<<m<<" "<<lay[k][m].in_value<<endl;
					int act=lay[k][m].act;
					float val;
					if(act==1)
					{
						val=_Sigmoid(lay[k][m].in_value);
					}
					else if(act==2)
					{		
						val=_tanh(lay[k][m].in_value);
					}
					else
					{						
						val=_ReLu(lay[k][m].in_value);
					}
					lay[k][m].value=val;
							
					//cout<<"hello"<<k<<m<<" "<<lay[k][m].value<<endl;							
				}
	
												
			}
			//display_neuron();
					
					
			//backpropogation	
					
			//output
			for(int k=0; k<output_no; k++)
			{
				//acc to act func
				//cout<<sol[j][k]<<endl;
				if(lay[layers-1][k].act==1)
				{
					lay[layers-1][k].delta=(sol[j]-lay[layers-1][k].value)*_dSigmoid(lay[layers-1][k].in_value);
				}
				else if(lay[layers-1][k].act==2)
				{
					lay[layers-1][k].delta=(sol[j]-lay[layers-1][k].value)*_dtanh(lay[layers-1][k].in_value);
				}
				else
				{
					lay[layers-1][k].delta=(sol[j]-lay[layers-1][k].value)*_dReLu(lay[layers-1][k].in_value);
				}
											
			}
			//change in weights for output layer
			for(int k=0; k<weights[layers-2].size(); k++)
			{
				for(int l=0; l<weights[layers-2][k].size(); l++)
				{
					dw[layers-2][k][l]=float(alpha*lay[layers-1][l].delta*lay[layers-1][l].value);
					//cout<<dw[layers-2][k][l]<<endl;
				}
			}
					
			//hidden layer
			//calculation of deltas and weights
			for(int k=layers-2; k>0; k--)
			{
				for(int l=0; l<lay[k].size(); l++)
				{
					float sum=0, val=0;
					for(int m=0; m<weights[k].size(); m++)
					{
						for(int n=0; n<weights[k][m].size(); n++)
						{
							sum+=lay[k+1][n].delta*weights[k][m][n];
						}
					}
					if(lay[k][l].act==1)
					{
						val=_dSigmoid(lay[k][l].in_value);
					}
					else if(lay[k][l].act==2)
					{
						val=_tanh(lay[k][l].in_value);
					}
					else
					{
						val=_dReLu(lay[k][l].in_value);							
					}
					lay[k][l].delta=sum+val;
					//cout<<lay[k][l].delta<<endl;
				}
			}
					
			for(int k=layers-2; k>=0; k--)
			{
				for(int l=0; l<dw[k].size(); l++)
				{
					for(int m=0; m<dw[k][l].size(); m++)
					{
						float temp=dw[k][l][m];
						dw[k][l][m]=alpha*lay[k+1][m].delta*lay[k][l].value-temp*fact;
					}					
				}
			}
					
					//updation of weights
			for(int k=0; k<weights.size(); k++)
			{
				for(int l=0; l<weights[k].size(); l++)
				{
					for(int m=0; m<weights[k][l].size(); m++)
					{
						weights[k][l][m]=weights[k][l][m]-dw[k][l][m] + decay*weights[k][l][m];
					}
				}
			}
			for(int k=0; k<weights.size(); k++)
			{
				for(int l=0; l<weights[k].size(); l++)
				{
					for(int m=0; m<weights[k][l].size(); m++)
					{
						while(weights[k][l][m]>10)	//normalize
							weights[k][l][m]=weights[k][l][m]-10;
					}
				}
			}
													
		}//one data entry loop ends
		//display_weight(weights);
		//display_weight(dw);			
		//testing on validations set
		sum=0;
		for(int j=0; j<valdata.size(); j++)
		{
		//for one input row
		//initialize
			for(int k=0; k<lay.size(); k++)
			{
				for(int l=0; l<lay[k].size(); l++)
				{
					lay[k][l].value=0;
					lay[k][l].in_value=0;
				}
			}
			//layer by layer calculation of values
			for(int k=0; k<valdata[0].size(); k++)
			{
				lay[0][k].in_value=valdata[j][k];
				lay[0][k].value=valdata[j][k];						
				//cout<<data[j][k]<<" ";
			}
			//cout<<endl;
			for(int k=1; k<layers; k++)
			{
				//find connections between them for layer k
				for(int m=0; m<weights[k-1].size(); m++)
				{
					for(int n=0; n<weights[k-1][m].size(); n++)
					{
						//neuron
						lay[k][n].in_value+=weights[k-1][m][n]*lay[k-1][m].value;							
					}
				}
				for(int m=0; m<lay[k].size(); m++)
				{
					//cout<<"hello"<<k<<m<<" "<<lay[k][m].in_value<<endl;
					int act=lay[k][m].act;
					float val;
					if(act==1)
					{
						val=_Sigmoid(lay[k][m].in_value);
					}	
					else if(act==2)
					{		
						val=_tanh(lay[k][m].in_value);
					}
					else
					{						
						val=_ReLu(lay[k][m].in_value);
					}
					lay[k][m].value=val;
					//cout<<"hello"<<k<<m<<" "<<lay[k][m].value<<endl<<endl;							
				}												
			}
			//calc error
			for(int k=0; k<lay[layers-1].size(); k++)
			{
				sum+=(lay[layers-1][k].value-valsol[j])*(lay[layers-1][k].value-valsol[j]);
			}	
		}
		curr_err=sum/2;
		cout<<"curr_err: "<<curr_err<<endl;
		if(err>curr_err)
		{
			for(int x=0; x<weights.size(); x++)
			{
				for(int y=0; y<weights[x].size(); y++)
				{
					for(int z=0; z<weights[x][y].size(); z++)
					{
						best_weights[x][y][z]=weights[x][y][z];
					}
				}
			}
		}
			
	}//iterations loop ends
	for(int i=0; i<weights.size(); i++)
	{
		for(int j=0; j<weights[i].size(); j++)
		{
			for(int k=0; k<weights[i][j].size(); k++)
			{
				weights[i][j][k]=best_weights[i][j][k];
			}
		}
	}
	return ret;
}//func ends

void NeuralNetwork::test(vector <vector <float > > data, vector <float> sol)
{
	float miss=0;			
	cout<<endl<<"============================================="<<endl;
	cout<<"Testing "<<endl;
	cout<<endl<<"============================================="<<endl;
			
	display_weight(weights);
	
	//initialize
	for(int j=0; j<data.size(); j++)
	{
		//for one input row
		//initialize
		for(int k=0; k<lay.size(); k++)
		{
			for(int l=0; l<lay[k].size(); l++)
			{
				lay[k][l].value=0;
				lay[k][l].in_value=0;
			}
		}
		//layer by layer calculation of values
		for(int k=0; k<data[0].size(); k++)
		{
			lay[0][k].in_value=data[j][k];
			lay[0][k].value=data[j][k];						
			//cout<<data[j][k]<<" ";
		}
		//cout<<endl;
		for(int k=1; k<layers; k++)
		{
			//find connections between them for layer k
			for(int m=0; m<weights[k-1].size(); m++)
			{
				for(int n=0; n<weights[k-1][m].size(); n++)
				{
					//neuron
					lay[k][n].in_value+=weights[k-1][m][n]*lay[k-1][m].value;							
				}
			}
			for(int m=0; m<lay[k].size(); m++)
			{
				//cout<<"hello"<<k<<m<<" "<<lay[k][m].in_value<<endl;
				int act=lay[k][m].act;
				float val;
				if(act==1)
				{
					val=_Sigmoid(lay[k][m].in_value);
				}
				else if(act==2)
				{		
					val=_tanh(lay[k][m].in_value);
				}
				else
				{						
					val=_ReLu(lay[k][m].in_value);
				}
				lay[k][m].value=val;
				//cout<<"hello"<<k<<m<<" "<<lay[k][m].value<<endl<<endl;							
			}												
		}
		//classifying values obtained into classes
		for(int k=0; k<lay[layers-1].size(); k++)
		{
			int class_out;
			if((lay[layers-1][k].value) > 0.2 )
			{
				class_out=-1;
			}
			else if((lay[layers-1][k].value) < 0.2  && (lay[layers-1][k].value)>0.2 )
			{
				class_out=1;
			}
			else
			{
				class_out=0;
			}
			float t=lay[layers-1][k].value;
			t*=(-1);
			cout<<"VALUE "<<t<<"  class: "<<class_out<<"   SOL IS "<<sol[j]<<endl;
			if(class_out!=sol[j])
			{
				miss++;
			}
		}	
	}//one data entry loop ends	
	cout<<endl<<"==============================================="<<endl;
	float accuracy=(data.size()-miss)/data.size();
	accuracy=accuracy*100;
	cout<<"Accuracy = "<<accuracy;
	cout<<endl<<"==============================================="<<endl;	
}//func ends
		
int main()
{
	int hid_layers;
	vector <int> neurons;
	vector <int> act_func;
	vector< vector <float> > dataset, data;
	vector < float >  sol;
	vector< vector <float > > train_data, test_data, val_data, train_d;
    vector <float > train_output, test_output, val_output, train_o;
    
    data=read_file();
    int input_no=data[0].size()-1;
	//cout<<"FILE READ"<<endl;
	
	//spliting data from file into input and output
	for(int i=0; i<data.size(); i++)
    {
        vector <float> v;
        for(int j=0; j<input_no; j++)
        {
        	v.push_back(data[i][j]);
		}
		dataset.push_back(v);
    }
    int l=dataset[0].size()-1;
    for(int i=0; i<dataset.size(); i++)
    {
        sol.push_back(data[i][input_no]);
    }
    int size=dataset.size();
    int rows_train=(int)size*RATIO;
    int rows_test=size-rows_train;
    
    //spliting data into test data and train data
    
    train_data=split_input(dataset, 0, rows_train, input_no);
    test_data=split_input(dataset, rows_train, size-rows_train, input_no);
    train_output=split_output(sol, 0, rows_train);
    test_output=split_output(sol, rows_train, size-rows_train);

	cout<<"Enter no. of hidden layers: "<<endl;
	cin>>hid_layers;
	cout<<endl;
	cout<<"Enter no. of neurons in each layer: "<<endl;
	for(int i=0; i<hid_layers; i++)
	{
		int n;
		cout<<"Neurons for hidden layer "<<i+1<<": ";
		cin>>n;
		neurons.push_back(n);
	}
	cout<<endl;
	cout<<"List of activation functions: "<<endl;
	cout<<"1. Sigmoid"<<endl;
	cout<<"2. tanh"<<endl;
	cout<<"3. ReLu"<<endl;
	cout<<endl;
	cout<<"Activation functions (Enter 1, 2 or 3)"<<endl;
	for(int i=0; i<hid_layers; i++)
	{
		int n;
		string str;
		cout<<"Activation function for hidden layer "<<i+1<<": ";
		cin>>n;
		
		act_func.push_back(n);
	}
	NeuralNetwork nn;
	int flag=1;
	cout<<endl<<"Choose type of architecture: "<<endl;
	cout<<"1. Fully connected "<<endl;
	cout<<"2. Self-customized network"<<endl;
	cin>>flag;
	if(flag==2)
	{
		int res;
		neurons.push_back(0);
		vector <vector <vector <float> > > self;
		for(int i=0; i<hid_layers-1; i++)
		{
			vector <vector <float > > temp;
			for(int j=0; j<neurons[i]; j++)
			{
				vector <float> t;
				for(int k=0; k<neurons[i+1]; k++)
				{
					//cout<<"Neuron "<<i+1<<j<<" with "<<i+2<<k<<": "<<endl;
					t.push_back(0);
				}
				temp.push_back(t);
			}
			self.push_back(temp);
		}
		neurons.pop_back();
		if(hid_layers>1)
		{
			cout<<"Connections press 1 to connect else press 0: "<<endl;
			for(int i=0; i<self.size(); i++)
			{
				cout<<"Connect layer "<<i<<" with: "<<endl;
				for(int j=0; j<self[i].size(); j++)
				{
					for(int k=0; k<self[i][j].size(); k++)
					{
						cout<<"Neuron "<<i+1<<j<<" with "<<i+2<<k<<": ";
						cin>> res;
						if(res==1)
						{
							self[i][j][k]=1;
						}
					}
				}
			}
		}
		cout<<endl<<"CREATING CONNECTIONS: "<<endl;
		nn.my_connections(hid_layers, self, act_func, input_no, neurons);
	}
	else
	{
		nn.fully_connected(input_no, hid_layers, neurons, act_func);
	}
	nn.weight_creation(input_no, neurons);
	
	float factor=0, decay=0, th, thr;
	int iterations, cflag=0;
	
	
	//adding momentum for quick convergence
	cout<<"Do you want to add momentum to avoid local minima? (Press 1 or 0)): "<<endl;
	cin>>flag;
	if(flag==1)
	{
		cout<<"Enter the factor: (Generally less than 0.5): ";
		cin>>factor;
	}
	else
	{
		factor=0;
	}
	cout<<"Do you want to add weight decay? (Press 1 or 0)): "<<endl;
	cin>>flag;
	if(flag==1)
	{
		cout<<"Enter the factor: (Generally less than 0.5): ";
		cin>>decay;
	}
	else
	{
		factor=0;
	}
	cout<<endl<<"Do you want to use ?"<<endl<<"1. Condition on iterations"<<endl<<"2. Cross-validation"<<endl<<"3. k-fold cross-validation"<<endl<<endl;
	cin>>flag;
	if(flag==2)
	{
		cout<<"Enter threshold for stopping condition: ";
		cin>>thr;
		
		float x=(float) rand()/RAND_MAX;
		int sizet=train_data.size();
		int rows=sizet*x;
		
		//printdata(train_data);
		train_d=split_input(train_data, 0, rows, input_no);
    	val_data=split_input(train_data, rows, sizet-rows, input_no);
    	train_o=split_output(sol, 0, rows_train);
    	val_output=split_output(sol, rows_train, sizet-rows);
    	
    	//printdata(train_d);
    	//printsol(train_o);
    	//printdata(val_data);
    	//printsol(val_output);
		
		nn.train_crossval(train_d, train_o, val_data, val_output, factor, decay, thr);
	}
	else if(flag==3)
	{
		int avg, ret, k;
		cout<<"Enter threshold for stopping condition: ";
		cin>>thr;
		cout<<"Enter no of times to perform cross-validation: ";
		cin>>k;
		for(int i=0; i<k; i++)
		{
			//initialize weights
			
			float x=(float) rand()/RAND_MAX;
			int sizet=train_data.size();
			int rows=sizet*x;
			nn.initialize_weights();
			train_d=split_input(train_data, 0, rows, input_no);
    		val_data=split_input(train_data, rows, sizet-rows, input_no);
    		train_o=split_output(sol, 0, rows_train);
    		val_output=split_output(sol, rows_train, sizet-rows);
    		
			ret=nn.train_crossval(train_d, train_o, val_data, val_output, factor, decay, thr);
			//cout<<"ret: "<<ret<<endl;
			avg+=ret;	
		}
		avg=avg/k;
		th=0.1;
		avg=(int)avg;
		cout<<"Average iterations: "<<avg<<endl;
		nn.train(train_data, train_output, avg, factor, decay, th);
		
	}
	else
	{
		flag=0;
		cout<<endl<<"Enter configurable parameters to avoid overfitting and regularization: "<<endl;
		cout<<"Do you want constant iterations or choose a threshold? (Press 1 or 0)";
		cin>>flag;
		if(flag==1)
		{
			cout<<"Enter no of iterations: ";
			cin>>iterations;
			th=0;
		}
		else
		{
			iterations=itr;
			cout<<"Enter threshold: ";
			cin>>th;
			cflag=1;
		}
		nn.train(train_data, train_output, iterations, factor, decay, th);
	}
	
	//nn.test(test_data, test_output);
	nn.test(train_data, train_output);
	
	
	cout<<endl<<endl<<"SUCCESS END";
	return 0;
}    
