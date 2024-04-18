

let x_vals = [];
let y_vals = [];

let a, b, c, d;

let Dragging = false;

const learningRate = 0.25; 
const optimizer = tf.train.adam(learningRate);  //adam=  instead of using in polynomial regression we used adam which is also a optimizer which also does slowly adjusting mnb to minimize the loss function




function setup(){
    createCanvas(600, 600);
//    background(colour = 'black');

a = tf.variable(tf.scalar(random(-1, 1))); //it would be a single number and will be  random
b = tf.variable(tf.scalar(random(-1, 1)));
c = tf.variable(tf.scalar(random(-1, 1)));
d = tf.variable(tf.scalar(random(-1, 1)));


}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
    
    //(pred, labels) => pred.sub(labels).square().mean();
}


function predict(x) {   // i'm just getting a plain array of numbers and turned them into a tensor and applied formula on it then

    const xs = tf.tensor1d(x);
    // y = ax^3+bx^2+cx+d
    const ys = xs.pow(3).mul(a)
    .add(xs.square(b))
    .add(xs.mul(c)
    .add(d));
 
    return ys;
}
function mousePressed () {
    Dragging = true;
}
function mouseReleased () {
    Dragging = false;

}

//function mouseDragged (){
    
//}
function draw() {
if (Dragging) {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
}
else {
    tf.tidy(() => {
    if (x_vals.length > 0){
    const ys = tf.tensor1d(y_vals);
    optimizer.minimize(() => loss(predict(x_vals), ys))
    }
    });
    }
    background(0);

    stroke(255);
    strokeWeight(4);
    for (let i = 0; i < x_vals.length; i++) {

        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], -1, 1, height, 0);
         point(px, py );
         
    
    }
// going for visulization
const curveX = [];
for (let x = -1; x < 1.02; x += 0.05){
    curveX.push(x);

}
   // console.log(curveX);    

const ys = tf.tidy(() => predict(curveX));
let  curveY = ys.dataSync(); 
//ys.print();
ys.dispose();

beginShape();
noFill();
stroke(255);
strokeWeight(4);
for ( let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);  
    vertex(x, y);

}
endShape();



//let x1 = map(curveX[0], -1, 1, 0, width);
//let x2 = map(curveX[1], -1, 1, 0, width);

//let y1 = map(curveY[0], -1, 1, height, 0);
//let y2 = map(curveY[0], -1, 1, height, 0);

//strokeWeight(2);
//line(x1, y1, x2, y2);

//ys.print();
//xs.dispose();
//ys.dispose();

}