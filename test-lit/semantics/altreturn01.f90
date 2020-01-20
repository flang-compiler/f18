! Check calls with alt returns

!RUN: %test_error %s %flang

       CALL TEST (N, *100, *200 )
       PRINT *,'Normal return'
       STOP
100    PRINT *,'First alternate return'
       STOP
200    PRINT *,'Secondnd alternate return'
       END
