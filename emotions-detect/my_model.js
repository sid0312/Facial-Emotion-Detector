let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new emotionsData();
var smiles=0, horns=0, frowns=0, upwards=0, typing=0;
let isPredicting = false;

async function loadMobilenet() 
{
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() 
{
  dataset.ys = null;
  dataset.encodeLabels(5);
    
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 5, activation: 'softmax'})
    ]
  });

  const optimizer =  tf.train.adam(0.0001)

 model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
 
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);

        console.log('Loss: ' + loss );
        
        }
      }
   });
}

function handleButton(elem){
	switch(elem.id){
		case "0":
			smiles++;
			document.getElementById("smiles").innerText = "Times you get happy: " + smiles;
			break;
		case "1":
			horns++;
			document.getElementById("horns").innerText = "Angry face count: " + horns;
			break;
		case "2":
			frowns++;
			document.getElementById("frowns").innerText = "Number of Frowns: " + frowns;
			break;  
		case "3":
			upwards++;
			document.getElementById("upwards").innerText = "Times you look upwards: " + upwards;
			break;
        case "4":
            typing++;
            document.getElementById("typing").innerText = "Typing samples: " + typing;
            break;
            
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "You're smiling";
			break;
		case 1:
			predictionText = "You're angry";
			break;
		case 2:
			predictionText = "You're frowning";
			break;
		case 3:
			predictionText = "You're looking upwards";
			break;
        case 4:
			predictionText = "You're typing something";
			break;
	
            
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
	alert("Training Done!")
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();