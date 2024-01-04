from metaflow import FlowSpec, step

class BranchFlow(FlowSpec):

    @step
    def start(self):
        self.contador = 0
        self.next(self.suma)

    @step
    def suma(self):
        print(f"El contador es {self.contador}")
        self.contador += 1
        self.next(self.rama_a, self.rama_b)

    @step
    def rama_a(self):
        self.x = 1
        self.next(self.union) 

    @step
    def rama_b(self):
        self.x = 2
        self.next(self.union) 

    @step
    def union(self, inputs):
        print(inputs.rama_a.x)
        print(inputs.rama_b.x)
        # En este momento, dentro de `inputs` tenemos dos versiones del atributo `contador`
        # `self.contador` ya no está definido por esta misma razón en el flow principal, 
        # para recuperarlo es importante usar `self.merge_artifacts`. 
        self.merge_artifacts(inputs, exclude=['x'])
        self.contador += 1
        self.next(self.end)

    @step
    def end(self):
        print(f"El contador final es {self.contador}")
        print("Final")
        pass


if __name__ == "__main__":
    BranchFlow()
