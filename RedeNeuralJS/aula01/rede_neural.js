function sigmoid(x){
    return 1 / (1+ Math.exp(-x));
}

function dsigmoid(x){
    return x * (1-x);
}

class Rede_neural{
    constructor(i_nodes, h_nodes, o_nodes){
        this.i_nodes = i_nodes;
        this.h_nodes = h_nodes;
        this.o_nodes = o_nodes;

        this.bias_ih = new Matriz(this.h_nodes, 1);
        this.bias_ih.randomize();

        this.bias_ho = new Matriz(this.o_nodes, 1);
        this.bias_ho.randomize();
        // this.bias_ih.print();
        // this.bias_ho.print();
        this.weigths_ih = new Matriz(this.h_nodes, this.i_nodes);
        this.weigths_ih.randomize();

        this.weigths_ho = new Matriz(this.o_nodes, this.h_nodes);
        this.weigths_ho.randomize();
        //this.weigths_ho.print();
        //this.weigths_ih.print();
    }

    feedforward(arr){
        // INPUT -> HIDDEN

        let input = Matriz.arrayToMatriz(arr);
        let hidden = Matriz.multiply(this.weigths_ih, input);
        hidden = Matriz.add(hidden, this.bias_ih);

        hidden.map(sigmoid);

        //HIDDEN -> OUTPUT

        let output = Matriz.multiply(this.weigths_ho, hidden);
        output = Matriz.add(output, this.bias_ho);
        output.map(sigmoid);
        output.print();
    }
}