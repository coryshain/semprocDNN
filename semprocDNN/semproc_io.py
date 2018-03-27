def dump_model(m, path, model_name, input_names, output_names, session):
    with session.as_default():
        with session.graph.as_default():
            with open(path, 'w') as f:
                W = m.eval(session=session)
                for i, a in enumerate(input_names):
                    for j, b in enumerate(output_names):
                        w = W[i,j]
                        f.write('%s %s : %s = %s\n' %(model_name, a, b, w))
