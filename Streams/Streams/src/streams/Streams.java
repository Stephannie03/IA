/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package streams;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;



/**
 *
 * @author Stephannie
 */
public class Streams {

    /**
     * @param args the command line arguments
     */
    
    //Ruta del archivo words
    static String file = "/Users/Stephannie/Documents/GitHub/IA/Streams/Streams/src/streams/words.txt";
    
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
       int charas = charas("casa"); 
       System.out.println("Cantidad de palabras con los mismos charas " + charas);
       
       int parte = parte("casa");
       System.out.println("Cantidad de palabras que son parte de otra " + parte);
       
       int vocales = vocales("hola");
       System.out.println("Cantidad de palabras que tienen igual sus vocales " + vocales);
    }
    
    public static int charas(String palabra) throws IOException{
        Stream<String> stream = Files.lines(Paths.get(file));
        List<String> cantidad = stream
                            .filter((String line) -> Arrays.asList(line.split(""))
                            .containsAll(Arrays.asList(palabra.split(""))))
                            .collect(Collectors.toList());
        return cantidad.size();


        
    }
    
    
    public static int parte(String palabra) throws IOException{
        Stream<String> stream = Files.lines(Paths.get(file));
        List<String> cantidad = stream
                .filter(line -> line.contains(palabra))
                .collect(Collectors.toList());
        return cantidad.size();

       
    }
    
    
    public static int vocales(String palabra) throws IOException{
        final String vocales = palabra.replaceAll("[bcdfghjklmnñpqrstvwxyz]", "");
       
        Stream<String> stream = Files.lines(Paths.get(file));
        List<String> cantidad = stream
                .filter((String line) -> Arrays.asList((line.replaceAll("[bcdfghjklmnñpqrstvwxyz]", "")).split(""))
                .containsAll(Arrays.asList(vocales.split(""))))
                .collect(Collectors.toList());
        return cantidad.size();

       
    }
    
  
    
}



   

