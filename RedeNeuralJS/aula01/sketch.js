//interface gráfica para o navegador utilizando canvas
//chamaremos a rede neural aqui!
function setup(){
    createCanvas(500, 500);
    background(0);

    var nn = new Rede_neural(1,3,5);
    var arr = [1,2];
    nn.feedforward(arr)

}
function draw(){

}