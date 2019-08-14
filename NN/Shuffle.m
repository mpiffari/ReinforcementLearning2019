 function mat_shuffled = Shuffle(mat, row, column)
     indexes = randi(row,row,1)';
     mat_shuffled = zeros(row,(column+1));
     for i = 1:row
         new_row = mat(indexes(i),:);
         mat_shuffled(i,:)= new_row;
     end
 end

