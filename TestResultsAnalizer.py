import pandas as pd
import xlrd
from semantics import SemanticComparer

class TestResult:
    def __init__(self,execution=None,suite=None,title=None,status=None,error=None):
        self.__execution = execution
        self.__suite = suite
        self.__title = title
        self.__status = status
        self.__error = error
        self.__bugs = []
        self.compareEngine = SemanticComparer()
        self.compareEngine.Reload()

    def hasPassed(self):
        if self.__status == 1:
            return True
        return False
    
    def getTestName(self):
        return self.__suite+"/"+self.__title
    
    def getError(self):
        return self.__error
    
    def getExecution(self):
        return self.__execution
    
    def addBug(self,bugId):
        self.__bugs.extend(bugId)
    
    def getBugs(self):
        return self.__bugs

class TestResultsAnalizer:
    def __init__(self, resultsfile):
        self.resultsfile = resultsfile
        self.testResults = []
        self.failingTests = []
        self.bugs = []

    def LoadTestResults(self):
        wb = xlrd.open_workbook(self.resultsfile)
        sheet = wb.sheet_by_index(0)
        for index in sheet.nrows:
            execution = sheet.cell_value(index, 0)
            suite = sheet.cell_value(index, 1)
            title = sheet.cell_value(index, 2)
            status = sheet.cell_value(index, 3)
            error = sheet.cell_value(index, 4)
            self.testResults.append(
                TestResult(
                    execution=execution,
                    suite=suite,
                    title=title,
                    status=status,
                    error=error
                )
            )
        print(str(len(self.testResults))+" tests were loaded")

    def FilterFailedTests(self):
        for test in self.testResults:
            if not test.hasPassed():
                self.failingTests.append(test)
        print(str(len(self.failingTests))+" failed tests were found")
    
    def AnalyzeTests(self):
        self.FilterFailedTests()
        bugIndex = 0
        for test in self.failingTests:
            index = 1
            if len(test.getBugs) > 0:
                bug = "Bug "+str(bugIndex)




