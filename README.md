# MLFlow Examples For ADSP

| Example                              | Description                                                                        |
|--------------------------------------|------------------------------------------------------------------------------------|
| background_job_example               | MLFlow Background Job On Anaconda Enterprise                                       |
| real_esrgan                          | Super Resolution With Real ESRGran - MLFlow Driven Parallelism within ADSP         |
| stable_diffusion                     | Stable Diffusion - MLFlow Driven Parallelism within ADSP                           |


### How To Deploy An Example

```commandline
anaconda-project run ae5 project upload --name {EXAMPLE NAME} {LOCAL PATH TO EXAMPLE}
```

For a specific example the command would look like these:
```commandline
anaconda-project run ae5 project upload --name real_esrgan examples/real_esrgan
```
