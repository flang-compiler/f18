! Check calls with alt returns

! RUN: %flang -fdebug-resolve-names -fparse-only %s 2>&1


       CALL TEST (N, *100, *200 )
       PRINT *,'Normal return'
       STOP
100    PRINT *,'First alternate return'
       STOP
200    PRINT *,'Secondnd alternate return'
       END
