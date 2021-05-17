import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/enums/dtype.dart';

import 'package:mongol/mongol.dart';

void main() => runApp(MyApp());

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  Model _customModel;
  TextEditingController _controller;

  // variable used to share the auto completed word between two functions:
  // 1). _sample and 2). runCustomModel
  List<dynamic> _tcac;
  // placeholder for auto generated words
  Set<String> _prediction;

  // mappings from token to characters and vise versa.
  Map<String, dynamic> _char_to_token;
  Map<int, dynamic> _token_to_char;

  // configuration variables
  int _blockSize; // maximum context to look back when running model inference.
  int _numberOfSample; // max number of attempts for generating words.
  int _lengthOfSample; // max length of a word
  int _maxNumberSampling; // max number of words to generate
  
  // random number generator. It is used for sampling next character given a 
  // probability distribution of characters.
  var rng = new Random();

  @override
  Future<void> initState() {
    super.initState();
    // setup configuration variables
    _blockSize = 20;
    _numberOfSample = 5;
    _lengthOfSample = 20;
    _maxNumberSampling = 10;

    _controller=new TextEditingController(text: 'ᡴᡭᢚ');
    // ignore: deprecated_member_use
    _prediction = new Set<String>();

    print('loading model');
    loadModel();

    print('loading mappings');
    loadMappings();
  }

  //load your model
  Future loadModel() async {
    String pathCustomModel = "assets/models/model.pt";
    try {
      _customModel = await PyTorchMobile.loadModel(pathCustomModel);
    } on PlatformException {
      print("only supported for android and ios so far");
    }
  }

  Future loadMappings() async {
    String pathMappings = "assets/mappings/char_to_token.json";
    try {
      String jsonString = await rootBundle.loadString(pathMappings);
      _char_to_token = jsonDecode(jsonString);
      _token_to_char = _char_to_token.map( (k, v) => MapEntry(v, k));
      for( var i = 0; i <87; i++ ) {
        var char = _token_to_char[i] as String;
        print(char);
      }

    } on PlatformException {
      print("only supported for android and ios so far");
    }
  }

  // softmax function. It turns a list of number into a probability
  // distribution which sum to one.
  // Definition:
  //    softmax([x1,x2,...,xn]) = [
  //      exp(x1)/(exp(x1)+exp(x2)+...+exp(xn)),
  //      exp(x2)/(exp(x1)+exp(x2)+...+exp(xn)),
  //      ...
  //      exp(xn)/(exp(x1)+exp(x2)+...+exp(xn))]
  List _softmax(List X) {
    var XExp = X.map((xi) => exp(xi)).toList();
    var sum = XExp.reduce((double a,double b) => a + b);
    List probs = XExp.map((xexpi) => (xexpi/sum)).toList();
    return probs;
  }

  //Generate one word given context.
  // The tokenContent is a list of number, each of which maps to certain
  // Mongolian basic character, e.g. 'ᠠ','ᠯ'. The mapping is defined in the
  // assets/mappings/char_to_token.json file.
  Future _sample(List tokenContext, int wordMaxLength) async{
    // deep copy context to avoid accumulation
    List x = new List.from(tokenContext);
    // cast from int to double. This step could be removed for a better code
    // quality
    x = x.map((i) => (i.toDouble() as double)).toList();

    // generate one character at a time until 'space' or 'new line' character
    // are generated.
    for( var i = 1; i <=wordMaxLength; i++ ) {
      // xCond is introduce to make sure the context only look back maximum
      // _blockSize number of characters.
      List xCond = x;
      if (x.length >_blockSize) {
        xCond = x.sublist(x.length - _blockSize,x.length);
      }

      // interact with the model and get the logits. Logits has shape
      // [1, xCond.length, _char_to_token.length] and it includes the raw number
      // that can be turned into probability for generating next character.
      var logits = await _customModel.getPrediction(xCond, [1,xCond.length], DType.int64) as List;

      // only consider the last slide of the logits. Others are logits based on
      // shorter previous context
      logits = logits.sublist((xCond.length-1) * _char_to_token.length, xCond.length * _char_to_token.length);

      // turn the logits into Probability Mass Function (PMF)
      List probs = await _softmax(logits);

      // get the Cumulative Probability Mass Function (CMF). It is needed since I
      // don't know how to sample from a distribution in dart. The idea here is
      // to build a sampling tool based on a known CMF.
      probs[0] = probs[0] as double;
      for( var j = 1; j <probs.length; j++ ) {
        probs[j] = (probs[j-1] + probs[j]) as double;
      }

      double randNum = rng.nextDouble();
      num ix = probs.indexWhere((p) => (p>randNum) as bool);

      // append the sampled new character to the context
      x.add(ix.toDouble());

      // if sampled new character is 'space' or 'new line', it means we obtain a
      // complete word already and we stop the generation.
      if(ix==1 || ix==0){
        break;
      }
    }

    // share the new generated word with other functions through _tcac.
    _tcac = x;
  }

  // Generate _maxNumberSampling number of auto completed words given the
  // context.
  Future runCustomModel() async {
    // refresh the old auto completions
    _prediction = new Set<String>();

    // get the context from the textField
    String context = _controller.text;

    // tokenize the characters into integer numbers, the mapping are defined in
    // assets/mappings/char_to_token.json file.
    List tokenizedContext = context.split('').map((ch) => (_char_to_token[ch] as int)).toList();
    tokenizedContext = tokenizedContext.map((i) => (i.toDouble() as double)).toList();

    // generate _numberOfSample number of auto completed words by trying at most
    // _maxNumberSampling times.
    for( var k = 0; k <_maxNumberSampling && _prediction.length<_numberOfSample; k++ ) {
      await _sample(tokenizedContext, _lengthOfSample);

      // construct the word by mapping back the tokens into the characters and
      // join the characters into a string.
      _prediction.add(_tcac.map((token) => (_token_to_char[token] as String))
          .toList()
          .join());
    }
    print('all auto completed words: $_prediction');
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(fontFamily: 'z52chimegtig'),
      home: Scaffold(
        appBar: AppBar(
          title: const Text(' ZMongol:\n Input Auto Completion Demo'),
        ),
        body: Column(
          //mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Container(
              child: Center(
                child: SizedBox(
                  width:200,
                  height:200,
                  child: MongolTextField(
                    controller: _controller,
                    decoration:null,
                    maxLines:null,
                  ),
                ),
              ),
        ),
            Container(
              child: TextButton(
                onPressed: runCustomModel,
                child: Text(
                  "Run custom model",
                  style: TextStyle(
                    color: Colors.black,
                  ),
                ),
              ),
            ),
            Container(
              child: Visibility(
                visible: _prediction != null,
                child: MongolText(_prediction.join('\n'),

                  style: TextStyle(fontFamily: 'z52chimegtig',
                  fontSize: 24),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

}