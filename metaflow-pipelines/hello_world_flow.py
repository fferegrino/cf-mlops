from metaflow import FlowSpec, step


class HelloWorldFlow(FlowSpec):
    @step
    def start(self):
        print("Primer paso")
        self.next(self.hello)

    @step
    def hello(self):
        print("Hola mundo")
        self.next(self.end)

    @step
    def end(self):
        print("Último paso")


if __name__ == "__main__":
    HelloWorldFlow()
