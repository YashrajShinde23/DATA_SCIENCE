#pgm to read entire content from text file

with open ("pi_digits.txt","r") as File:
    content=File.read()
    print(content.rstrip())




#pgm to read first n lines from text file
with open("pi_digits.txt","r") as file:
    lines=file.readlines()
    if lines:
        print("first line:",lines[0].strip())
        print("last line:",lines[-1].strip())





#pgm to accept input from user and append it
filename="c:/1-python/programming.text"
with open(filename,"w") as file:
    file.write("i love programming./n")
    file.write("i love creating new games./n")
    in_line=input("enter the line")
    file.write(in_line)




#pgm to read contents from text file line by line
#and store  each line in list
filename="c:/1-python/pi_digits.txt"
with open(filename,"r") as file:
    lines=file.readlines()
    pi_string=[]
    for line in lines:
        pi_string.append(line.rstrip())
        print(pi_string)
    print(len(pi_string))    



#pgm to find longest word from text file

filename="c:/1-python/programming.text"
with open(filename,"r") as file:
    lines=file.readlines()
    longest_word=""
    for line in lines:
        words=line.split()
        for word in words:
            if(len(word)>len(longest_word)):
                longest_word=word
print("the longest word:",longest_word)



#pgm to find smallest word from text file

filename="c:/1-python/programming.text"
with open(filename,"r") as file:
    lines=file.readlines()
    smallest_word=None
    
    for line in lines:
        words=line.split()
        for word in words:
            if smallest_word==None or ((len(word)<len(smallest_word))):
                smallest_word=word
print("the smallest word:",smallest_word)




#pgm to count the frequency of user entered word in text
filename="c:/1-python/programming.text"
input_line=input("enter the text:")
words=input_line.split()
word_count=len(words)
with open(filename,"w") as file:
    file.write(input_line)
print("the total words entered:",word_count)


                  

    






